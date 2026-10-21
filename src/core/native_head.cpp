#include "strata/core/native_head.hpp"
#include "conversation_store_sha256.hpp"
#include "native_verified_upload.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>
#include <climits>
#include <cstdio>
#include <cstring>
#include <exception>
#include <utility>

namespace {
using strata::core::conversation_store_detail::sha256;
using strata::core::conversation_store_detail::sha256_hex;
// Only verified loads stage/hash. Each upload reads the SAME bounded immutable
// host chunk that entered SHA-256, rather than hashing a mapping then reopening it.
cudaError_t upload_native(void* destination, const void* source, uint64_t bytes,
                          bool verify, std::string& digest) {
    if (!verify) return cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice);
    cudaError_t status = cudaSuccess;
    const bool complete = strata::core::native_verified_upload(source, size_t(bytes), digest,
        [&](size_t offset, std::span<const std::byte> chunk) {
            status = cudaMemcpy(static_cast<unsigned char*>(destination) + offset, chunk.data(), chunk.size(),
                                cudaMemcpyHostToDevice);
            return status == cudaSuccess;
        });
    return complete ? cudaSuccess : (status == cudaSuccess ? cudaErrorInvalidValue : status);
}
struct UploadOwner {
    void* weights = nullptr;
    void* scratch = nullptr;
    ~UploadOwner() { if (scratch) cudaFree(scratch); if (weights) cudaFree(weights); }
};
} // namespace

namespace strata::core {

NativeHead::~NativeHead() {
    unload();
}

void NativeHead::unload() {
    if (weights_) strata::kernels::native_q6_k_unpack(weights_);
    if (scratch_) cudaFree(scratch_);
    if (weights_) cudaFree(weights_);
    scratch_ = nullptr;
    weights_ = nullptr;
    identity_ = {};
}

bool NativeHead::load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, std::string& err,
                      bool verify_identity) {
    if (loaded()) { err = "native head is already loaded"; return false; }
    if (n_in <= 0 || n_out <= 0 || n_in > INT_MAX || n_out > INT_MAX || n_in % 256) {
        err = "native head requires positive int32 dimensions and whole 256-value rows";
        return false;
    }
    try {
        // The architecture is the metadata shard's; output.weight comes from whichever shard holds it (shard 2
        // of Unsloth's UD-Q4_K_XL, whose shard 1 holds no tensor).  GgufModel refuses a duplicate across shards.
        const strata::GgufModel model(shards);
        err = strata::check_engine_architecture(model.meta());
        if (!err.empty()) return false;
        size_t at = 0;
        const strata::TensorInfo* tensor = model.find("output.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!tensor || !strata::kernels::native_mmvq_supported((int) tensor->type) || tensor->shape.size() != 2 ||
            tensor->shape[0] != (uint64_t) n_in || tensor->shape[1] != (uint64_t) n_out) {
            err = "native head: expected a natively supported output.weight with the canonical head dimensions";
            return false;
        }
        const uint64_t bytes = strata::kernels::native_mmvq_weight_bytes((int) tensor->type, (int) n_in, (int) n_out);
        const uint64_t payload = gguf.file_size() - gguf.data_start();
        if (tensor->offset > payload || bytes > payload - tensor->offset) {
            err = "native head: truncated output.weight payload";
            return false;
        }
        NativeTensorIdentity identity;
        if (verify_identity) {
            identity.source = native_source_identity(gguf.path());
            identity.generation = next_native_weight_generation();
        }
        UploadOwner upload;
        cudaError_t status = cudaMalloc(&upload.weights, bytes);
        if (status == cudaSuccess)
            status = cudaMalloc(&upload.scratch, strata::kernels::native_q8_1_bytes((int) n_in, 1));
        strata::platform::advise_willneed(gguf.tensor_data(*tensor), bytes);
        if (status == cudaSuccess)
            status = upload_native(upload.weights, gguf.tensor_data(*tensor), bytes, verify_identity, identity.sha256);
        if (status != cudaSuccess) {
            err = std::string("native head upload: ") + cudaGetErrorString(status);
            return false;
        }
        if (verify_identity && identity.source != native_source_identity(gguf.path())) {
            err = "native head source changed during verified upload";
            return false;
        }
        int device = -1;
        const auto device_status = cudaGetDevice(&device);
        if (device_status == cudaSuccess) device_ = device;
        if (verify_identity && (device_status != cudaSuccess || device_ < 0)) {
            err = "native head cannot identify its owning device";
            return false;
        }
        weights_ = upload.weights; upload.weights = nullptr;
        scratch_ = upload.scratch; upload.scratch = nullptr;
        bytes_ = bytes;
        n_in_ = (int) n_in;
        n_out_ = (int) n_out;
        type_ = (int) tensor->type;
        if (verify_identity) {
            identity.bytes = bytes_; identity.input = n_in_; identity.output = n_out_;
            identity.type = type_; identity.device = device_;
            identity_ = std::move(identity);
        }
        if (type_ == 14 && strata::kernels::native_q6_k_packed_enabled())   // STRATA_Q6_PACKED=1
            strata::kernels::native_q6_k_pack(weights_, n_in_, n_out_, "output head");
        return true;
    } catch (const std::exception& error) {
        err = std::string("native head: ") + error.what();
        if (verify_identity) unload();
        return false;
    }
}

size_t NativeHead::projection_scratch_bytes(int ncols) const {
    return n_in_ > 0 ? strata::kernels::native_q8_1_bytes(n_in_, ncols) : 0;
}
bool NativeHead::run(const float* mixed, float* logits, void* stream, std::string& err) const {
    return run_with_scratch(mixed, logits, scratch_, projection_scratch_bytes(), stream, err);
}
bool NativeHead::run_with_scratch(const float* mixed, float* logits, void* scratch, size_t scratch_bytes,
                                  void* stream, std::string& err) const {
    return run_rows_with_scratch(mixed, logits, 1, scratch, scratch_bytes, stream, err);
}
bool NativeHead::run_rows_with_scratch(const float* mixed, float* logits, int ncols, void* scratch,
                                       size_t scratch_bytes, void* stream, std::string& err) const {
    if (!loaded() || !mixed || !logits || !stream || !scratch || ncols < 1 || ncols > 8 ||
        scratch_bytes < projection_scratch_bytes(ncols)) {
        err = "native head requires loaded weights, device buffers, bounded scratch and an explicit stream";
        return false;
    }
    try {
        if (type_ == 13) {
            strata::kernels::native_q5_k_f32(weights_, mixed, scratch, logits, n_in_, n_out_, ncols, stream);
        } else {
            strata::kernels::native_quantize_q8_1(mixed, scratch, n_in_, ncols, stream);
            strata::kernels::native_mmvq(type_, weights_, scratch, logits, n_in_, n_out_, ncols, stream);
        }
    } catch (const std::exception& error) {
        err = std::string("native head launch: ") + error.what();
        return false;
    }
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        err = std::string("native head launch: ") + cudaGetErrorString(status);
        return false;
    }
    return true;
}

// ================================ plan v0.3 P6: THE NATIVE EMBEDDING ================================

namespace {
const NativeEmbed* g_embed = nullptr;
}
void set_native_embed(const NativeEmbed* e) { g_embed = e; }
const NativeEmbed* native_embed() { return g_embed; }

NativeEmbed::~NativeEmbed() {
    if (host_) cudaFreeHost(host_);
    else if (dev_) cudaFree(const_cast<void*>(dev_));   // the VRAM fallback below
}

bool NativeEmbed::load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab, std::string& err,
                       bool verify_identity) {
    if (loaded()) { err = "native embedding is already loaded"; return false; }
    try {
        const strata::GgufModel model(shards);
        // --embd-gguf's one-tensor file (tools/embd_bf16_pack.py) says "strata-embd": only its tensor is checked
        const strata::MetaValue* arch = model.meta().get("general.architecture");
        err = arch != nullptr && arch->s == "strata-embd" ? std::string() : strata::check_engine_architecture(model.meta());
        if (!err.empty()) { err = "native embedding: " + err; return false; }
        size_t at = 0;
        const strata::TensorInfo* t = model.find("token_embd.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) n_embd || t->shape[1] != (uint64_t) n_vocab ||
            !strata::kernels::embed_type_supported((int) t->type) || n_embd % 256) {
            err = "native embedding: token_embd.weight is absent, of another shape, or of a type without a GPU "
                  "dequantizer";
            return false;
        }
        row_ = strata::kernels::iq_row_bytes((int) t->type, n_embd);
        bytes_ = (uint64_t) row_ * (uint64_t) n_vocab;
        // the table is copied out of the mapping below: a truncated shard must be an error, not a read past EOF
        if (!model.in_bounds(*t, at) || strata::tensor_payload_bytes(*t) != bytes_) {
            err = "native embedding: token_embd.weight's payload is truncated or not " + std::to_string(bytes_) +
                  " B (" + gguf.path() + ")";
            bytes_ = 0;
            return false;
        }
        strata::platform::advise_willneed(gguf.tensor_data(*t), bytes_);   // copied out of the mapping below
        NativeTensorIdentity identity;
        if (verify_identity) {
            identity.source = native_source_identity(gguf.path());
            identity.generation = next_native_weight_generation();
        }
        cudaError_t host_err = cudaHostAlloc(&host_, bytes_, cudaHostAllocMapped | cudaHostAllocPortable);
        if (host_err != cudaSuccess) {
            // Under WSL2 the driver's pinned/mapped host budget (~1 GiB) can be spent by the GPU contexts
            // themselves (three cards). The table is only gathered from, so keep it in the current device's VRAM
            // instead: it costs its size there and reads faster than over PCIe.
            cudaGetLastError();
            host_ = nullptr;
            void* d = nullptr;
            if (cudaMalloc(&d, bytes_) != cudaSuccess ||
                upload_native(d, gguf.tensor_data(*t), bytes_, verify_identity, identity.sha256) != cudaSuccess) {
                if (d) cudaFree(d);
                cudaGetLastError();
                err = "native embedding: cannot pin " + std::to_string(bytes_ >> 20) + " MiB: " +
                      cudaGetErrorString(host_err) + ", nor place it in VRAM";
                return false;
            }
            std::fprintf(stderr, "strata: native embedding: cannot pin %llu MiB, kept in VRAM instead\n",
                         (unsigned long long) (bytes_ >> 20));
            dev_ = d;
        } else {
            std::memcpy(host_, gguf.tensor_data(*t), bytes_);
            if (verify_identity) identity.sha256 = sha256_hex(sha256(host_, size_t(bytes_)));
            void* d = nullptr;
#if defined(STRATA_USE_HIP)
            // #325: the Windows HIP stack can refuse the device alias of a mapped allocation (and, when it gives
            // one, it is the host address itself - unified addressing; kernels read it correctly there, a
            // device-to-device copy into it does not land: tests/hip/mapped_alias). The table is only gathered from,
            // so without an alias it goes into VRAM like the unpinnable case above, instead of failing the start.
            if (cudaHostGetDevicePointer(&d, host_, 0) != cudaSuccess || d == nullptr) {
                cudaGetLastError();
                d = nullptr;
                if (cudaMalloc(&d, bytes_) != cudaSuccess ||
                    cudaMemcpy(d, verify_identity ? host_ : gguf.tensor_data(*t), bytes_, cudaMemcpyHostToDevice) != cudaSuccess) {
                    if (d) cudaFree(d);
                    cudaGetLastError();
                    err = "native embedding: no device alias for the mapped table, and no VRAM to copy it into";
                    return false;
                }
                cudaFreeHost(host_);   // the destructor frees dev_ when host_ is null
                host_ = nullptr;
                std::fprintf(stderr, "strata: native embedding: no device alias for the mapped table, kept in VRAM\n");
            }
#else
            if (cudaHostGetDevicePointer(&d, host_, 0) != cudaSuccess) {
                err = "native embedding: no device alias for the mapped table";
                return false;
            }
#endif
            dev_ = d;
        }
        type_ = (int) t->type;
        n_embd_ = n_embd;
        n_vocab_ = n_vocab;
        if (verify_identity) {
            if (identity.source != native_source_identity(gguf.path())) {
                err = "native embedding source changed during verified upload";
                if (host_) cudaFreeHost(host_); else if (dev_) cudaFree(const_cast<void*>(dev_));
                host_ = nullptr; dev_ = nullptr; bytes_ = 0;
                return false;
            }
            int device = -1;
            if (cudaGetDevice(&device) != cudaSuccess) {
                err = "native embedding cannot identify its owning device";
                if (host_) cudaFreeHost(host_); else if (dev_) cudaFree(const_cast<void*>(dev_));
                host_ = nullptr; dev_ = nullptr; bytes_ = 0;
                return false;
            }
            identity.bytes = bytes_; identity.input = n_embd_; identity.output = n_vocab_;
            identity.type = type_; identity.device = device;
            identity_ = std::move(identity);
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("native embedding: ") + e.what();
        if (verify_identity) {
            if (host_) cudaFreeHost(host_); else if (dev_) cudaFree(const_cast<void*>(dev_));
            host_ = nullptr; dev_ = nullptr; bytes_ = 0; identity_ = {};
        }
        return false;
    }
}

void NativeEmbed::gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const {
    strata::kernels::iq_embed_rows(type_, dev_, row_, tokens, n_tok, n_embd_, out, stream);
}

void NativeEmbed::gather_one(int64_t token, float* out, void* stream) const {
    strata::kernels::iq_dequant_f32(type_, (const uint8_t*) dev_ + (size_t) token * row_, n_embd_, out, stream);
}

}  // namespace strata::core
