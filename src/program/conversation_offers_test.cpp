#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "strata/program/conversation_offers.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace strata::program;
namespace {
int checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
#if defined(__linux__)
using Packet = std::array<uint8_t, 72>;
void put64(uint8_t* p, uint64_t value) { for (unsigned i = 0; i < 8; ++i) p[i] = uint8_t(value >> (8 * i)); }
void put32(uint8_t* p, uint32_t value) { for (unsigned i = 0; i < 4; ++i) p[i] = uint8_t(value >> (8 * i)); }
uint64_t get64(const uint8_t* p) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= uint64_t(p[i]) << (8 * i);
    return value;
}
uint64_t deadline() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count()) + 10000000000ULL;
}
Packet packet(uint16_t type, uint64_t id, uint64_t generation, uint32_t tokens = 0) {
    Packet out{};
    std::memcpy(out.data(), "SCC1", 4);
    out[4] = 1;
    out[6] = uint8_t(type);
    put64(out.data() + 8, id);
    put64(out.data() + 16, generation);
    put64(out.data() + 24, deadline());
    put32(out.data() + 32, tokens);
    put32(out.data() + 36, 1);
    out[40] = 91;
    return out;
}
void send_packet(int fd, const Packet& bytes, const std::vector<int>& fds = {}) {
    iovec iov{const_cast<uint8_t*>(bytes.data()), bytes.size()};
    alignas(cmsghdr) std::array<uint8_t, CMSG_SPACE(sizeof(int) * 2)> control{};
    msghdr message{};
    message.msg_iov = &iov;
    message.msg_iovlen = 1;
    if (!fds.empty()) {
        check(fds.size() <= 2, "fixture descriptor count bounded");
        message.msg_control = control.data();
        message.msg_controllen = CMSG_SPACE(fds.size() * sizeof(int));
        auto* c = CMSG_FIRSTHDR(&message);
        c->cmsg_level = SOL_SOCKET;
        c->cmsg_type = SCM_RIGHTS;
        c->cmsg_len = CMSG_LEN(fds.size() * sizeof(int));
        std::memcpy(CMSG_DATA(c), fds.data(), fds.size() * sizeof(int));
    }
    check(::sendmsg(fd, &message, MSG_NOSIGNAL) == ssize_t(bytes.size()), "send complete control packet");
}
int token_fd(const std::vector<int32_t>& ids, bool sealed = true, bool readonly = true) {
    const int fd = ::memfd_create("conversation-offer-test", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    check(fd >= 0, "create private token memfd");
    std::vector<uint8_t> bytes(ids.size() * 4);
    for (size_t i = 0; i < ids.size(); ++i) put32(bytes.data() + 4 * i, uint32_t(ids[i]));
    check(::write(fd, bytes.data(), bytes.size()) == ssize_t(bytes.size()), "write finite token payload");
    if (sealed)
        check(::fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) == 0,
              "seal token payload against every mutation");
    if (!readonly) return fd;
    char path[64];
    std::snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
    const int result = ::open(path, O_RDONLY | O_CLOEXEC);
    ::close(fd);
    check(result >= 0, "open read-only transfer descriptor");
    return result;
}
void offer(int peer, uint64_t id, uint64_t generation, const std::vector<int32_t>& ids) {
    const int fd = token_fd(ids);
    send_packet(peer, packet(2, id, generation, uint32_t(ids.size())), {fd});
    ::close(fd);
}
size_t descriptors() {
    DIR* directory = ::opendir("/proc/self/fd");
    check(directory != nullptr, "open descriptor inventory");
    size_t count = 0;
    while (const auto* entry = ::readdir(directory)) if (entry->d_name[0] != '.') ++count;
    ::closedir(directory);
    return count;
}
struct Observed {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<ConversationOffer> offers;
    std::vector<std::pair<uint64_t, uint64_t>> cancellations;
    bool take(ConversationOffer&& value) {
        std::lock_guard lock(mutex);
        offers.push_back(std::move(value));
        changed.notify_all();
        return true;
    }
    void cancel(uint64_t id, uint64_t generation) {
        std::lock_guard lock(mutex);
        cancellations.emplace_back(id, generation);
        changed.notify_all();
    }
    void await(size_t requests, size_t cancelled) {
        std::unique_lock lock(mutex);
        check(changed.wait_for(lock, std::chrono::seconds(3), [&] {
            return offers.size() >= requests && cancellations.size() >= cancelled;
        }), "bounded receiver reached expected request transition");
    }
};
Packet hello(int peer) {
    timeval timeout{3, 0};
    check(::setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout) == 0, "bound handshake read");
    Packet result{};
    check(::recv(peer, result.data(), result.size(), 0) == ssize_t(result.size()), "receive exact HELLO");
    check(std::memcmp(result.data(), "SCC1", 4) == 0 && result[4] == 1 && result[6] == 1,
          "HELLO protocol version and type are explicit");
    return result;
}
#endif
} // namespace

int main() {
#if !defined(__linux__)
    return 77;
#else
    const size_t original_fds = descriptors();
    {
        int socket[2];
        check(::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, socket) == 0, "create private channel");
        Observed observed;
        ConversationOfferChannel channel(socket[1], 4096, 1, 16384,
            [&](ConversationOffer&& value) { return observed.take(std::move(value)); },
            [&](uint64_t id, uint64_t generation) { observed.cancel(id, generation); });
        std::array<uint8_t, 32> key{};
        key[0] = 17;
        std::string why;
        check(channel.start(key, 9, true, why), "start valid channel");
        const auto greeting = hello(socket[0]);
        check(get64(greeting.data() + 16) == 9 && get64(greeting.data() + 24) == 16384 && greeting[32] == 1 &&
              greeting[36] == 1 && greeting[40] == 17, "HELLO binds issuer epoch and bounded frontier");
        offer(socket[0], 1, 7, {4, 90, 12003, 8});
        observed.await(1, 0);
        {
            std::lock_guard lock(observed.mutex);
            const auto& got = observed.offers.front();
            check(got.request_id == 1 && got.generation == 7 && got.scope.value[0] == 91 && got.cvec &&
                  got.tokens == std::vector<int32_t>({4, 90, 12003, 8}), "sealed handoff preserves exact authorized prompt");
        }
        send_packet(socket[0], packet(3, 1, 6)); // stale cancel cannot release generation 7
        offer(socket[0], 2, 1, {6});            // frontier still occupied
        send_packet(socket[0], packet(3, 1, 7));
        observed.await(1, 1);
        check(channel.stats().rejected == 1, "stale cancel cannot admit a follower past full frontier");

        // Every malformed transfer must close its received descriptors and leave
        // the original serving transport usable for a subsequent valid request.
        const size_t before_bad = descriptors();
        int fd = token_fd({1, 2}, false);
        send_packet(socket[0], packet(2, 3, 1, 2), {fd});
        ::close(fd);
        fd = token_fd({1, 2}, true, false);
        send_packet(socket[0], packet(2, 3, 1, 2), {fd});
        ::close(fd);
        fd = token_fd({1, 2});
        send_packet(socket[0], packet(2, 3, 1, 3), {fd}); // wrong byte extent
        send_packet(socket[0], packet(2, 3, 1, 2), {fd, fd});
        auto bad = packet(2, 3, 1, 2);
        bad[40] = 0;
        send_packet(socket[0], bad, {fd}); // absent authorization
        bad = packet(2, 3, 1, 2);
        put64(bad.data() + 24, 1);
        send_packet(socket[0], bad, {fd}); // expired original deadline
        bad = packet(2, 3, 1, 2);
        bad[4] = 99;
        send_packet(socket[0], bad, {fd});
        ::close(fd);
        fd = token_fd({-1});
        send_packet(socket[0], packet(2, 3, 1, 1), {fd});
        ::close(fd);
        offer(socket[0], 4, 2, {17, 18});
        send_packet(socket[0], packet(3, 4, 2));
        observed.await(2, 2);
        check(channel.stats().rejected == 9, "unsafe, malformed and expired offers are rejected");
        check(descriptors() == before_bad, "rejected received descriptors do not leak");

        offer(socket[0], 1, 8, {500});
        observed.await(3, 2);
        send_packet(socket[0], packet(3, 1, 7)); // late predecessor cancellation
        ::shutdown(socket[0], SHUT_WR);        // peer retirement cancels only generation 8
        observed.await(3, 3);
        channel.stop();
        {
            std::lock_guard lock(observed.mutex);
            check(observed.cancellations.back() == std::pair<uint64_t, uint64_t>(1, 8),
                  "peer exit retires the actual successor, not a stale generation");
        }
        check(channel.stats().accepted_token_bytes == 28, "only admitted token bytes counted");
        ::close(socket[0]);
    }
    check(descriptors() == original_fds, "complete channel teardown releases all descriptors");
    {
        int socket[2];
        check(::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, socket) == 0, "create byte-limited channel");
        Observed observed;
        ConversationOfferChannel channel(socket[1], 4096, 2, 16,
            [&](ConversationOffer&& value) { return observed.take(std::move(value)); },
            [&](uint64_t id, uint64_t generation) { observed.cancel(id, generation); });
        std::array<uint8_t, 32> key{};
        key[0] = 11;
        std::string why;
        check(channel.start(key, 2, true, why), "start byte-limited channel");
        hello(socket[0]);
        offer(socket[0], 1, 1, {1, 2, 3, 4});
        offer(socket[0], 2, 1, {5});
        send_packet(socket[0], packet(3, 1, 1));
        observed.await(1, 1);
        check(channel.stats().rejected == 1, "byte limit binds despite a free descriptor slot");
        offer(socket[0], 2, 1, {5});
        observed.await(2, 1);
        channel.stop();
        observed.await(2, 2);
        ::close(socket[0]);
    }
    {
        int socket[2];
        check(::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, socket) == 0, "create disabled channel");
        ConversationOfferChannel channel(socket[1], 4096, 1, 16384,
            [](ConversationOffer&&) { return false; }, [](uint64_t, uint64_t) {});
        std::array<uint8_t, 32> key{};
        key[0] = 99;
        std::string why;
        check(channel.start(key, 0, false, why), "disabled cache has an explicit handshake");
        const auto greeting = hello(socket[0]);
        check(greeting[36] == 0 && std::memcmp(greeting.data() + 40, std::array<uint8_t, 32>{}.data(), 32) == 0,
              "disabled persistence cannot disclose or activate an issuer key");
        channel.stop();
        ::close(socket[0]);
    }
    check(descriptors() == original_fds, "disabled and cancelled channels leave no descriptors");
    ::setenv("STRATA_CONVERSATION_OFFER_FD", "1", 1);
    check(inherited_conversation_offer_fd() == -1 && !std::getenv("STRATA_CONVERSATION_OFFER_FD"),
          "standard streams cannot be inherited as offers and environment ownership is consumed");
    std::printf("conversation_offers_test: %d checks passed\n", checks);
#endif
}
