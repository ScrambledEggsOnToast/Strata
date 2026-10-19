#pragma once

#include "strata/prefill/segments.hpp"
#include <chrono>
#include <exception>
#include <future>
#include <limits>
#include <string>
#include <utility>

namespace strata::prefill {

struct StagePacketShape {
    int producer_device = -1, consumer_device = -1;
    int64_t layer_begin = 0, layer_end = 0, consumer_layer_end = 0;
    int64_t position = 0, rows = 0, columns = 0;
    bool single_chunk = false, pipe_chunked = false;
};

// One physical handoff slot, not another transport. Only its submitting thread mutates
// it; a consumer borrows the immutable descriptor until its completion has joined.
// A failed/cancelled consumer is still a consumer: neither that outcome nor a failed
// producer permits early storage reuse. Drain refusal leaves the slot quarantined.
class StagePacket {
public:
    using Release = void (*)(float*);
    using ProducerDrain = bool (*)(void*, std::string&);
    enum class Outcome { idle, complete, cancelled, failed };

    StagePacket() = default;
    StagePacket(const StagePacket&) = delete;
    StagePacket& operator=(const StagePacket&) = delete;
    ~StagePacket() {
        std::string ignored;
        if (active_) drain(ignored);
        if (active_) std::terminate();
        if (payload_) release_(payload_);
    }

    // Adopt once at initialization. The production adapter supplies cudaFreeHost;
    // fixtures can observe the identical lifetime contract without a CUDA context.
    bool adopt(float* payload, size_t elements, Release release) {
        if (payload_ || active_ || !payload || !elements || !release) return false;
        payload_ = payload; capacity_ = elements; release_ = release;
        return true;
    }
    const void* transport() const { return producer_; }
    bool allocated() const { return payload_ != nullptr; }
    float* payload() const { return payload_; }
    const StagePacketShape& shape() const { return shape_; }
    const OwnerSegment* owners() const { return owners_.data(); }
    int owner_count() const { return owner_count_; }
    uint64_t generation() const { return generation_; }
    bool reusable() const { return !active_; }
    Outcome outcome() const { return outcome_; }
    std::string& consumer_error() { return error_; }

    bool begin(const StagePacketShape& shape, const OwnerSegment* owners, int count,
               void* producer, ProducerDrain drain, std::string& error) {
        if (active_ || !payload_ || !drain || generation_ == std::numeric_limits<uint64_t>::max() ||
            shape.rows <= 0 || shape.columns <= 0 || size_t(shape.rows) > capacity_ / size_t(shape.columns) ||
            shape.position < 0 || shape.layer_begin < 0 || shape.layer_end <= shape.layer_begin ||
            shape.consumer_layer_end <= shape.layer_end || !owners || count < 1 || count > kPrefillOwners) {
            error = "prefill packet: occupied slot or invalid shape/completion"; return false;
        }
        int64_t row = 0;
        for (int i = 0; i < count; ++i) {
            if (owners[i].row != row || owners[i].owner.rows <= 0 || owners[i].owner.rows > shape.rows - row) {
                error = "prefill packet: owner rows do not partition the payload"; return false;
            }
            row += owners[i].owner.rows;
        }
        if (row != shape.rows) { error = "prefill packet: incomplete owner rows"; return false; }
        shape_ = shape;
        std::copy_n(owners, count, owners_.data());
        owner_count_ = count;
        ++generation_;
        producer_ = producer; producer_drain_ = drain;
        active_ = true; published_ = false; producer_drained_ = false;
        consumer_started_ = consumer_drained_ = consumer_ok_ = false;
        failed_drain_ = false;
        outcome_ = Outcome::idle;
        error_.clear();
        return true;
    }

    bool publish(uint64_t generation, bool producer_drained, std::string& error) {
        if (!current(generation) || published_) {
            error = "prefill packet: stale or repeated producer publication"; return false;
        }
        producer_drained_ = producer_drained;
        published_ = true;
        return true;
    }
    bool accepts(uint64_t generation, int device, int64_t layer_begin, int64_t layer_end,
                 int64_t position, int64_t rows, int64_t columns) const {
        return current(generation) && published_ && shape_.consumer_device == device &&
            shape_.layer_end == layer_begin && shape_.consumer_layer_end == layer_end &&
            shape_.position == position && shape_.rows == rows && shape_.columns == columns;
    }
    bool attach(uint64_t generation, std::future<bool>&& completion, std::string& error) {
        if (!current(generation) || !published_ || consumer_started_ || !completion.valid()) {
            error = "prefill packet: stale or repeated consumer completion"; return false;
        }
        consumer_ = std::move(completion);
        consumer_started_ = true;
        return true;
    }
    // The synchronous retained path crosses the same completion interface.
    bool finish(uint64_t generation, bool success, std::string& error) {
        if (!current(generation) || !published_ || consumer_started_) {
            error = "prefill packet: stale or repeated inline completion"; return false;
        }
        consumer_started_ = consumer_drained_ = true;
        consumer_ok_ = success;
        if (!success) error_ = error;
        return true;
    }
    bool pending() const { return consumer_.valid(); }
    bool ready() const {
        return consumer_.valid() && consumer_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    // Always attempt both joins, even after an exception/failed producer. Repeated
    // drain is idempotent, and no failure is converted into a successful completion.
    bool drain(std::string& error) {
        if (!active_) {
            if (outcome_ == Outcome::failed || outcome_ == Outcome::cancelled) error = error_;
            return outcome_ == Outcome::idle || outcome_ == Outcome::complete;
        }
        if (consumer_.valid()) {
            try { consumer_ok_ = consumer_.get(); }
            catch (const std::exception& failure) { error_ = failure.what(); consumer_ok_ = false; }
            catch (...) { error_ = "prefill packet: consumer threw"; consumer_ok_ = false; }
            consumer_drained_ = true;
        } else if (!consumer_started_) {
            // A producer may fail after enqueue and before publishing/launching.
            consumer_drained_ = true;
            consumer_ok_ = false;
            if (error_.empty()) error_ = "prefill packet: producer did not hand off";
        }
        std::string producer_error;
        if (!producer_drained_) producer_drained_ = producer_drain_(producer_, producer_error);
        if (!producer_drained_) {
            error = producer_error.empty() ? "prefill packet: producer did not drain" : producer_error;
            error_ = error;
            failed_drain_ = true;
            outcome_ = Outcome::failed;
            return false;
        }
        if (!consumer_drained_) {
            error = "prefill packet: consumer has no completion owner";
            outcome_ = Outcome::failed;
            return false;
        }
        active_ = false;
        if (published_ && consumer_ok_ && !failed_drain_) outcome_ = Outcome::complete;
        else {
            if (error_.empty()) error_ = "prefill packet: consumer failed";
            outcome_ = error_ == "cancelled" ? Outcome::cancelled : Outcome::failed;
        }
        error = error_;
        return outcome_ == Outcome::complete;
    }

private:
    bool current(uint64_t generation) const { return active_ && generation && generation == generation_; }
    float* payload_ = nullptr;
    size_t capacity_ = 0;
    Release release_ = nullptr;
    StagePacketShape shape_;
    std::array<OwnerSegment, kPrefillOwners> owners_{};
    int owner_count_ = 0;
    uint64_t generation_ = 0;
    void* producer_ = nullptr;
    ProducerDrain producer_drain_ = nullptr;
    std::future<bool> consumer_;
    std::string error_;
    Outcome outcome_ = Outcome::idle;
    bool active_ = false, published_ = false, producer_drained_ = false;
    bool consumer_started_ = false, consumer_drained_ = false, consumer_ok_ = false;
    bool failed_drain_ = false;
};
} // namespace strata::prefill
