// No-load regression at the production recursive drain seam. Including its translation
// unit exposes the private Impl only to this friend fixture, without a runtime test
// hook or a second implementation of the drain/cancellation rules.
#include "../../src/prefill/prefill.cpp"

namespace strata::prefill {
struct PrefillDrainFixture {
    struct Producer {
        bool ready = true;
        int joins = 0;
        static bool drain(void* context, std::string& error) {
            auto& producer = *static_cast<Producer*>(context);
            ++producer.joins;
            if (!producer.ready) error = "fixture producer incomplete";
            return producer.ready;
        }
    } producers[2];
    // Reverse destruction keeps every successor alive through its parent's cleanup.
    Prefill tail, middle, root;
    std::function<bool(Prefill*, int64_t, std::string&)> step =
        [](Prefill*, int64_t, std::string&) { return true; };
    std::string error;
    PrefillDrainFixture() {
        root.set_stage(0, 20, &middle);
        middle.set_stage(20, 39, &tail);
        tail.set_stage(39, 48, nullptr);
        for (Prefill* stage : {&root, &middle, &tail}) {
            stage->retained_step_ = &step;
            stage->impl_->bound_request = 7;
            stage->impl_->retained_cancelled = true;
        }
    }
    ~PrefillDrainFixture() {
        // Even the failed-physical-drain fixture must finish its owned storage.
        for (auto& producer : producers) producer.ready = true;
        root.drain_pipeline(error);
    }
    void inline_completion(Prefill& stage, Producer& producer, const char* terminal,
                           bool producer_already_drained = true) {
        auto& packet = stage.impl_->handoff[0].packet;
        if (!packet.allocated() && !packet.adopt(new float[4]{}, 4, [](float* p) { delete[] p; }))
            std::abort();
        OwnerSegment owner;
        owner.owner.request = stage.impl_->bound_request;
        owner.owner.position = 0; owner.owner.rows = 1;
        StagePacketShape shape{0, 1, stage.stage_lb_, stage.stage_le_, stage.next_->stage_le_,
                               0, 1, 4, true, true};
        if (!packet.begin(shape, &owner, 1, &producer, Producer::drain, error) ||
            !packet.publish(packet.generation(), producer_already_drained, error)) std::abort();
        error = terminal;
        if (!packet.finish(packet.generation(), error.empty(), error)) std::abort();
        stage.next_packet_ = &packet;
    }
    void cancel_both() {
        // STOP was observed in the third stage after both upstream packets were
        // published. These are the two inline completion receipts left on unwind.
        inline_completion(middle, producers[1], "cancelled");
        inline_completion(root, producers[0], "cancelled");
    }
    void drain() { root.drain_pipeline(error); }
    bool idle() {
        std::string ignored;
        return root.context_idle(ignored);
    }
    bool all_reusable() const {
        return root.impl_->handoff[0].packet.reusable() && middle.impl_->handoff[0].packet.reusable() &&
               !root.next_packet_ && !middle.next_packet_;
    }
    bool both_poisoned() const { return root.impl_->poisoned && middle.impl_->poisoned; }
    uint64_t generation() const { return root.impl_->handoff[0].packet.generation(); }
};
} // namespace strata::prefill

namespace {
int failures = 0;
void check(bool condition, const char* name) {
    std::printf("  %-90s %s\n", name, condition ? "ok" : "FAIL");
    failures += !condition;
}
}
int main() {
    using strata::prefill::PrefillDrainFixture;
    std::puts("prefill_drain_test (production three-stage retained drain, no CUDA work)");
    {
        PrefillDrainFixture chain;
        chain.cancel_both();
        chain.drain();
        check(chain.error == "cancelled" && chain.all_reusable(),
              "three-stage inline cancellation drains both real production packet owners");
        check(chain.idle(), "clean final-stage cancellation does not poison the recursive retained ancestors");
        const auto generation = chain.generation();
        check(chain.root.bind_request(8, chain.error), "the same drained production chain accepts its next request binding");
        chain.cancel_both();
        chain.drain();
        check(chain.error == "cancelled" && chain.idle() && chain.all_reusable() &&
              chain.generation() == generation + 1,
              "a second cancellation reuses the joined generation without stale ownership");
    }
    {
        PrefillDrainFixture chain;
        chain.inline_completion(chain.middle, chain.producers[1], "fixture consumer failed");
        chain.inline_completion(chain.root, chain.producers[0], "cancelled");
        chain.drain();
        check(chain.both_poisoned() && chain.all_reusable() && !chain.idle(),
              "real descendant failure still poisons retained ancestors after physical joins");
        check(chain.error == "fixture consumer failed", "a genuine descendant failure dominates upstream cancellation");
    }
    {
        PrefillDrainFixture chain;
        chain.inline_completion(chain.middle, chain.producers[1], "", false);
        chain.inline_completion(chain.root, chain.producers[0], "cancelled");
        chain.producers[1].ready = false;
        chain.drain();
        check(chain.both_poisoned() && !chain.all_reusable() && !chain.idle() && chain.producers[1].joins > 0,
              "an incomplete producer drain retains storage and poisons every retained ancestor");
        check(chain.error == "fixture producer incomplete", "physical drain failure cannot be relabelled as cancellation");
        chain.producers[1].ready = true;
        chain.drain();
        check(chain.all_reusable() && chain.both_poisoned() && !chain.idle(),
              "later physical cleanup does not erase the genuine failed-owner quarantine");
    }
    std::printf("prefill_drain_test: %s (%d failures)\n", failures ? "FAILED" : "all passed", failures);
    return failures ? 1 : 0;
}
