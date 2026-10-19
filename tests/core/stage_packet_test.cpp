// Header-only lifetime fixtures. The same completion interface owns production CUDA handoffs.
#include "strata/prefill/handoff.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using strata::prefill::OwnerSegment;
using strata::prefill::StagePacket;
using strata::prefill::StagePacketShape;
namespace {
int failures = 0, releases = 0;
void check(bool value, const char* name) {
    std::printf("  %-82s %s\n", name, value ? "ok" : "FAIL");
    failures += !value;
}
void release(float* payload) { ++releases; delete[] payload; }
struct Producer {
    int joins = 0;
    bool permit = true;
    static bool drain(void* context, std::string& error) {
        auto& producer = *static_cast<Producer*>(context);
        ++producer.joins;
        if (!producer.permit) error = "producer not drained";
        return producer.permit;
    }
};
struct Fixture {
    Producer producer;
    StagePacket packet;
    StagePacketShape shape{0, 1, 0, 20, 39, 8, 2, 4, false, true};
    OwnerSegment owner;
    std::string error;
    Fixture() {
        if (!packet.adopt(new float[8]{1, 2, 3, 4, 5, 6, 7, 8}, 8, release)) std::abort();
        owner.owner.request = 7; owner.owner.incarnation = 3;
        owner.owner.slot = 1; owner.owner.position = 8; owner.owner.rows = 2;
    }
    bool begin() { return packet.begin(shape, &owner, 1, &producer, Producer::drain, error); }
    bool publish(bool drained = false) { return packet.publish(packet.generation(), drained, error); }
    bool finish(bool success = true) { return packet.finish(packet.generation(), success, error); }
};

#if defined(__GNUC__)
__attribute__((noinline))
#endif
uint64_t round_trip(Fixture& f) {
    if (!f.begin() || !f.publish(true) || !f.finish() || !f.packet.drain(f.error)) std::abort();
    return f.packet.generation() + f.packet.owners()[0].owner.request;
}
#if defined(__GNUC__)
__attribute__((noinline))
#endif
uint64_t empty_round(uint64_t value) { return value + 7; }
void benchmark() {
    Fixture f;
    constexpr int count = 200000;
    uint64_t sum = 0;
    auto first = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) sum += empty_round(uint64_t(i) + 1);
    auto second = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) sum += round_trip(f);
    auto third = std::chrono::steady_clock::now();
    const double empty = std::chrono::duration<double, std::nano>(second - first).count() / count;
    const double packet = std::chrono::duration<double, std::nano>(third - second).count() / count;
    std::printf("{\"kind\":\"packet_bookkeeping_microbenchmark\",\"iterations\":%d,"
                "\"empty_call_ns\":%.6f,\"packet_round_ns\":%.6f,\"net_ns\":%.6f,"
                "\"packet_bytes\":%zu,\"owner_segment_bytes\":%zu,\"checksum\":%llu}\n",
                count, empty, packet, packet - empty, sizeof(StagePacket), sizeof(OwnerSegment),
                (unsigned long long)sum);
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--benchmark") { benchmark(); return 0; }
    std::puts("stage_packet_test");
    {
        Fixture f;
        check(f.begin() && f.publish(), "producer publishes an immutable request/generation/shape descriptor");
        const auto generation = f.packet.generation();
        check(f.packet.accepts(generation, 1, 20, 39, 8, 2, 4) &&
              !f.packet.accepts(generation, 0, 20, 39, 8, 2, 4) &&
              !f.packet.accepts(generation, 1, 20, 39, 9, 2, 4), "wrong consumer device or shape is refused");
        check(!f.begin() && f.packet.owners()[0].owner.request == 7, "live storage cannot be reused or have its identity replaced");
        check(f.finish() && f.packet.drain(f.error) && f.producer.joins == 1 && f.packet.reusable(),
              "successful consumer plus producer join makes storage reusable");
        check(f.packet.drain(f.error) && f.producer.joins == 1, "repeated drain does not double-join or release storage");
        f.owner.owner.request = 8; f.owner.owner.incarnation = 4;
        check(f.begin() && f.publish() && f.packet.generation() == generation + 1 &&
              f.packet.owners()[0].owner.request == 8 && f.packet.owners()[0].owner.incarnation == 4,
              "slot reuse installs the next request and seat incarnation");
        check(!f.packet.accepts(generation, 1, 20, 39, 8, 2, 4) &&
              !f.packet.finish(generation, true, f.error), "late old-generation consumer cannot complete the reused slot");
        std::promise<bool> stale;
        auto completion = stale.get_future();
        check(!f.packet.attach(generation, std::move(completion), f.error) && completion.valid(),
              "refused stale completion remains owned by its caller");
        stale.set_value(true);
        check(f.finish() && f.packet.drain(f.error), "a rejected stale completion does not poison the current generation");
    }
    {
        Fixture f;
        f.begin(); f.publish();
        std::promise<void> allow, entered;
        auto gate = allow.get_future();
        const int before = releases;
        std::atomic<bool> copied{false};
        auto completion = std::async(std::launch::async, [&] {
            entered.set_value(); gate.wait();
            copied.store(f.packet.payload()[7] == 8 && f.packet.owners()[0].owner.request == 7);
            return true;
        });
        entered.get_future().wait();
        check(f.packet.attach(f.packet.generation(), std::move(completion), f.error) &&
              f.packet.pending() && !f.packet.ready() && !f.packet.reusable() && releases == before,
              "a delayed consumer retains its payload and completion owner");
        allow.set_value();
        check(f.packet.drain(f.error) && copied && f.producer.joins == 1 && releases == before,
              "consumer observes original bytes before physical storage can be recycled");
    }
    {
        Fixture f;
        f.begin(); f.publish();
        f.error = "cancelled"; f.finish(false);
        check(!f.packet.drain(f.error) && f.packet.reusable() && f.producer.joins == 1 &&
              f.packet.outcome() == StagePacket::Outcome::cancelled,
              "cancel after enqueue still joins producer and reports cancellation, not success");
    }
    {
        Fixture f;
        f.begin(); // enqueued producer fails before publication; no consumer exists
        check(!f.packet.drain(f.error) && f.packet.reusable() && f.producer.joins == 1 &&
              f.error == "prefill packet: producer did not hand off", "producer failure before consumer launch still drains its writes");
    }
    {
        Fixture f;
        f.begin(); f.publish();
        std::promise<bool> failed;
        auto completion = failed.get_future();
        f.packet.attach(f.packet.generation(), std::move(completion), f.error);
        failed.set_exception(std::make_exception_ptr(std::runtime_error("consumer failed")));
        check(!f.packet.drain(f.error) && f.packet.reusable() && f.producer.joins == 1 &&
              f.error == "consumer failed", "throwing consumer cannot skip producer drain or lose its failure");
    }
    {
        Fixture f;
        f.begin(); f.publish(); f.finish(); f.producer.permit = false;
        const int before = releases;
        check(!f.packet.drain(f.error) && !f.packet.reusable() && !f.begin() && releases == before,
              "partial drain quarantines storage even though the consumer completed");
        f.producer.permit = true;
        check(!f.packet.drain(f.error) && f.packet.reusable() && f.producer.joins == 2 &&
              f.packet.outcome() == StagePacket::Outcome::failed,
              "later physical cleanup frees the quarantine but cannot erase the failed verdict");
    }
    {
        Fixture parent, tail;
        parent.begin(); parent.publish(); parent.error = "consumer failed after forwarding"; parent.finish(false);
        tail.begin(); tail.publish(); tail.finish(); tail.producer.permit = false;
        const bool head_ok = parent.packet.drain(parent.error);
        const bool tail_ok = tail.packet.drain(tail.error); // the production chain also never short-circuits
        check(!head_ok && !tail_ok && parent.producer.joins == 1 && tail.producer.joins == 1 &&
              !tail.packet.reusable(), "failed direct successor does not imply its partially drained descendant is reusable");
        tail.producer.permit = true; tail.packet.drain(tail.error);
    }
    {
        Fixture f;
        auto owners = std::array<OwnerSegment, 2>{f.owner, f.owner};
        owners[0].owner.rows = 1; owners[1].row = 1; owners[1].owner.rows = 1;
        owners[1].owner.request = 9; owners[1].owner.incarnation = 5;
        check(f.packet.begin(f.shape, owners.data(), 2, &f.producer, Producer::drain, f.error) && f.publish() &&
              f.packet.owner_count() == 2 && f.packet.owners()[1].owner.request == 9,
              "packed owner identities partition one shared physical payload without becoming its owners");
        f.finish(); f.packet.drain(f.error);
    }
    {
        Fixture f;
        f.shape.rows = 3;
        check(!f.begin() && f.packet.generation() == 0 && f.packet.reusable(), "oversized shape is refused before ownership or generation mutation");
        f.shape.rows = 2; f.owner.row = 1;
        check(!f.begin() && f.packet.generation() == 0, "incomplete owner partition is refused before mutation");
    }
    const int before = releases;
    {
        Fixture f;
        f.begin(); f.publish();
        std::promise<bool> done;
        f.packet.attach(f.packet.generation(), done.get_future(), f.error);
        done.set_value(true);
    }
    check(releases == before + 1, "destruction joins outstanding completions and releases each payload exactly once");
    std::printf("stage_packet_test: %s (%d failures)\n", failures ? "FAILED" : "all passed", failures);
    return failures ? 1 : 0;
}
