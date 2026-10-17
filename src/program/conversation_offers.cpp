#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "strata/program/conversation_offers.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

namespace strata::program {
namespace {
constexpr size_t header_bytes = 72;
constexpr uint16_t protocol_version = 1;
constexpr uint16_t hello_type = 1, offer_type = 2, cancel_type = 3;

#if defined(__linux__)
uint16_t load16(const uint8_t* p) { return uint16_t(p[0]) | uint16_t(uint16_t(p[1]) << 8); }
uint32_t load32(const uint8_t* p) {
    uint32_t n = 0;
    for (unsigned i = 0; i < 4; ++i) n |= uint32_t(p[i]) << (8 * i);
    return n;
}
uint64_t load64(const uint8_t* p) {
    uint64_t n = 0;
    for (unsigned i = 0; i < 8; ++i) n |= uint64_t(p[i]) << (8 * i);
    return n;
}
void store64(uint8_t* p, uint64_t n) {
    for (unsigned i = 0; i < 8; ++i) p[i] = uint8_t(n >> (8 * i));
}
void store32(uint8_t* p, uint32_t n) {
    for (unsigned i = 0; i < 4; ++i) p[i] = uint8_t(n >> (8 * i));
}
uint64_t monotonic_ns() {
    timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 || now.tv_sec < 0) return 0;
    return uint64_t(now.tv_sec) * 1000000000ULL + uint64_t(now.tv_nsec);
}
struct ReceivedFds {
    std::array<int, 16> values{};
    size_t count = 0;
    ~ReceivedFds() { for (size_t i = 0; i < count; ++i) ::close(values[i]); }
};
#endif
} // namespace

int inherited_conversation_offer_fd() {
#if defined(__linux__)
    const char* text = std::getenv("STRATA_CONVERSATION_OFFER_FD");
    if (!text) return -1;
    // Copy the small decimal value before removing its backing environment entry.
    int fd = -1;
    if (*text >= '0' && *text <= '9') {
        char* end = nullptr;
        errno = 0;
        const long value = std::strtol(text, &end, 10);
        if (!errno && end && !*end && value >= 3 && value <= INT_MAX) fd = int(value);
    }
    ::unsetenv("STRATA_CONVERSATION_OFFER_FD");
    return fd;
#else
    return -1;
#endif
}

struct ConversationOfferChannel::Impl {
    int fd;
    size_t max_tokens, frontier, metadata_bytes;
    Offer offer;
    Cancel cancel;
    std::atomic<bool> stopping{false};
    bool started = false, enabled = false;
    std::thread reader;
    std::atomic<uint64_t> accepted{0}, cancelled{0}, rejected{0}, token_bytes{0}, cancel_failed{0};
    struct Active { uint64_t id = 0, generation = 0; size_t bytes = 0; };
    std::array<Active, 2> active{};
    size_t outstanding_bytes = 0;

    Impl(int fd_, size_t max_tokens_, size_t frontier_, size_t metadata_bytes_, Offer offer_, Cancel cancel_)
        : fd(fd_), max_tokens(max_tokens_), frontier(frontier_), metadata_bytes(metadata_bytes_),
          offer(std::move(offer_)), cancel(std::move(cancel_)) {}

    void cancel_at(size_t i) noexcept {
        const Active previous = active[i];
        if (!previous.id) return;
        try {
            cancel(previous.id, previous.generation);
        } catch (...) {
            // Store work may still be live: retain its slot and byte charge.
            cancel_failed.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        active[i] = {};
        outstanding_bytes -= previous.bytes;
        cancelled.fetch_add(1, std::memory_order_relaxed);
    }

#if defined(__linux__)
    bool payload(int source_fd, uint32_t count, std::vector<int32_t>& tokens) {
        const size_t bytes = size_t(count) * sizeof(int32_t);
        struct stat st{};
        const int access = ::fcntl(source_fd, F_GETFL);
        const int seals = ::fcntl(source_fd, F_GET_SEALS);
        constexpr int required = F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL;
        if (access < 0 || (access & O_ACCMODE) != O_RDONLY || seals < 0 || (seals & required) != required ||
            ::fstat(source_fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 || uint64_t(st.st_size) != bytes)
            return false;
        tokens.resize(count);
        size_t at = 0;
        while (at < bytes && !stopping.load(std::memory_order_relaxed)) {
            const ssize_t n = ::pread(source_fd, reinterpret_cast<uint8_t*>(tokens.data()) + at,
                                      bytes - at, off_t(at));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            at += size_t(n);
        }
        if (at != bytes) return false;
        // The wire is little-endian even on a differently ordered native host.
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian) != 1)
            for (auto& token : tokens) {
                const uint32_t value = load32(reinterpret_cast<const uint8_t*>(&token));
                std::memcpy(&token, &value, sizeof token);
            }
        return std::all_of(tokens.begin(), tokens.end(), [](int32_t token) { return token >= 0; });
    }

    void receive() noexcept {
        while (!stopping.load(std::memory_order_relaxed)) {
            std::array<uint8_t, header_bytes + 1> packet{};
            alignas(cmsghdr) std::array<uint8_t, CMSG_SPACE(sizeof(int) * 16)> control{};
            iovec iov{packet.data(), packet.size()};
            msghdr message{};
            message.msg_iov = &iov;
            message.msg_iovlen = 1;
            message.msg_control = control.data();
            message.msg_controllen = control.size();
            const ssize_t n = ::recvmsg(fd, &message, MSG_CMSG_CLOEXEC);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            ReceivedFds received;
            bool ancillary_ok = true;
            for (cmsghdr* c = CMSG_FIRSTHDR(&message); c; c = CMSG_NXTHDR(&message, c)) {
                if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS || c->cmsg_len < CMSG_LEN(0)) {
                    ancillary_ok = false;
                    continue;
                }
                const size_t data_bytes = c->cmsg_len - CMSG_LEN(0);
                if (data_bytes % sizeof(int)) ancillary_ok = false;
                for (size_t j = 0; j + sizeof(int) <= data_bytes; j += sizeof(int)) {
                    int received_fd = -1;
                    std::memcpy(&received_fd, CMSG_DATA(c) + j, sizeof received_fd);
                    if (received.count < received.values.size()) received.values[received.count++] = received_fd;
                    else { ::close(received_fd); ancillary_ok = false; }
                }
            }
            auto refuse = [&] { rejected.fetch_add(1, std::memory_order_relaxed); };
            if (n != ssize_t(header_bytes) || (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) || !ancillary_ok ||
                std::memcmp(packet.data(), "SCC1", 4) || load16(packet.data() + 4) != protocol_version) {
                refuse();
                continue;
            }
            const uint16_t type = load16(packet.data() + 6);
            const uint64_t id = load64(packet.data() + 8), generation = load64(packet.data() + 16);
            if (!id || (id & (uint64_t{1} << 63)) || !generation) { refuse(); continue; }
            if (type == cancel_type) {
                if (received.count) { refuse(); continue; }
                for (size_t i = 0; i < frontier; ++i)
                    if (active[i].id == id && active[i].generation == generation) { cancel_at(i); break; }
                continue;
            }
            const uint64_t deadline = load64(packet.data() + 24), now = monotonic_ns();
            const uint32_t count = load32(packet.data() + 32), flags = load32(packet.data() + 36);
            const uint64_t bytes = uint64_t(count) * sizeof(int32_t);
            core::ConversationScope scope;
            std::memcpy(scope.value.data(), packet.data() + 40, scope.value.size());
            size_t free_slot = frontier;
            bool duplicate = false;
            for (size_t i = 0; i < frontier; ++i) {
                duplicate |= active[i].id == id;
                if (!active[i].id && free_slot == frontier) free_slot = i;
            }
            if (!enabled || type != offer_type || received.count != 1 || !count || count > max_tokens ||
                flags > 1 || !scope.valid() || !now || deadline <= now || duplicate || free_slot == frontier ||
                outstanding_bytes > metadata_bytes || bytes > metadata_bytes - outstanding_bytes) {
                refuse();
                continue;
            }
            try {
                ConversationOffer request;
                request.request_id = id;
                request.generation = generation;
                request.deadline_ns = deadline;
                request.scope = scope;
                request.cvec = (flags & 1) != 0;
                if (!payload(received.values[0], count, request.tokens) || !offer(std::move(request))) {
                    refuse();
                    continue;
                }
                active[free_slot] = {id, generation, size_t(bytes)};
                outstanding_bytes += size_t(bytes);
                accepted.fetch_add(1, std::memory_order_relaxed);
                token_bytes.fetch_add(bytes, std::memory_order_relaxed);
            } catch (...) {
                // A failed enqueue cannot leave this request reusable. This is
                // cancellation of optional read work, never a GPU-state fallback.
                try { cancel(id, generation); } catch (...) {}
                refuse();
            }
        }
        for (size_t i = 0; i < frontier; ++i) cancel_at(i);
    }
#endif
};

ConversationOfferChannel::ConversationOfferChannel(int fd, size_t max_tokens, size_t frontier,
                                                 size_t metadata_bytes, Offer offer, Cancel cancel)
    : impl_(std::make_unique<Impl>(fd, max_tokens, frontier, metadata_bytes, std::move(offer), std::move(cancel))) {}

ConversationOfferChannel::~ConversationOfferChannel() { stop(); }

bool ConversationOfferChannel::start(const std::array<uint8_t, 32>& issuer_key, uint64_t epoch,
                                    bool enabled, std::string& why) {
    auto& p = *impl_;
    if (p.started) { why = "conversation offer channel already started"; return false; }
    p.started = true;
#if defined(__linux__)
    int kind = 0;
    socklen_t length = sizeof kind;
    if (p.fd < 3 || !p.max_tokens || p.max_tokens > UINT32_MAX || !p.frontier || p.frontier > p.active.size() ||
        !p.metadata_bytes || !p.offer || !p.cancel || ::getsockopt(p.fd, SOL_SOCKET, SO_TYPE, &kind, &length) ||
        kind != SOCK_SEQPACKET || ::fcntl(p.fd, F_SETFD, FD_CLOEXEC) < 0) {
        why = "invalid conversation offer channel or limits";
        return false;
    }
    if (enabled && (!epoch || std::none_of(issuer_key.begin(), issuer_key.end(), [](uint8_t b) { return b != 0; }))) {
        why = "conversation offer channel requires an enabled issuer epoch";
        return false;
    }
    std::array<uint8_t, header_bytes> hello{};
    std::memcpy(hello.data(), "SCC1", 4);
    hello[4] = uint8_t(protocol_version);
    hello[6] = uint8_t(hello_type);
    store64(hello.data() + 16, epoch);
    store64(hello.data() + 24, p.metadata_bytes);
    store32(hello.data() + 32, uint32_t(p.frontier));
    store32(hello.data() + 36, enabled ? 1 : 0);
    if (enabled) std::memcpy(hello.data() + 40, issuer_key.data(), issuer_key.size());
    ssize_t sent;
    do { sent = ::send(p.fd, hello.data(), hello.size(), MSG_NOSIGNAL | MSG_DONTWAIT); } while (sent < 0 && errno == EINTR);
    if (sent != ssize_t(hello.size())) { why = "conversation offer HELLO could not be delivered"; return false; }
    p.enabled = enabled;
    if (enabled) {
        try { p.reader = std::thread([&p] { p.receive(); }); }
        catch (const std::exception&) { why = "conversation offer reader could not start"; return false; }
    }
    return true;
#else
    (void) issuer_key; (void) epoch; (void) enabled;
    why = "persistent conversation offers require Linux";
    return false;
#endif
}

void ConversationOfferChannel::stop() noexcept {
    if (!impl_) return;
    auto& p = *impl_;
    p.stopping.store(true, std::memory_order_relaxed);
#if defined(__linux__)
    if (p.fd >= 3) ::shutdown(p.fd, SHUT_RDWR);
#endif
    if (p.reader.joinable()) p.reader.join();
#if defined(__linux__)
    if (p.fd >= 3) { ::close(p.fd); p.fd = -1; }
#endif
}

ConversationOfferStats ConversationOfferChannel::stats() const noexcept {
    const auto& p = *impl_;
    return {p.accepted.load(std::memory_order_relaxed), p.cancelled.load(std::memory_order_relaxed),
            p.rejected.load(std::memory_order_relaxed), p.token_bytes.load(std::memory_order_relaxed),
            p.cancel_failed.load(std::memory_order_relaxed)};
}

} // namespace strata::program
