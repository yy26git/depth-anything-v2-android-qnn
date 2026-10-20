#include <jni.h>
#include <onnxruntime_c_api.h>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const OrtApi& api() {
    const OrtApi* value = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!value) throw std::runtime_error("Pinned ORT C API is unavailable");
    return *value;
}
void checked(OrtStatus* status) {
    if (!status) return;
    const std::string message = api().GetErrorMessage(status);
    api().ReleaseStatus(status);
    throw std::runtime_error(message);
}
struct ValueDeleter { void operator()(OrtValue* value) const { if (value) api().ReleaseValue(value); } };
struct MemoryDeleter { void operator()(OrtMemoryInfo* value) const { if (value) api().ReleaseMemoryInfo(value); } };
using Value = std::unique_ptr<OrtValue, ValueDeleter>;
using Memory = std::unique_ptr<OrtMemoryInfo, MemoryDeleter>;

std::vector<int64_t> shape(JNIEnv* env, jlongArray dimensions) {
    const auto length = env->GetArrayLength(dimensions);
    if (length <= 0) throw std::runtime_error("UINT16 tensor has no shape");
    std::vector<jlong> java_dimensions(length);
    env->GetLongArrayRegion(dimensions, 0, length, java_dimensions.data());
    return {java_dimensions.begin(), java_dimensions.end()};
}
size_t bytes(const std::vector<int64_t>& dimensions) {
    size_t result = sizeof(uint16_t);
    for (const int64_t dimension : dimensions) {
        if (dimension <= 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max() / result)
            throw std::runtime_error("Invalid UINT16 tensor extent");
        result *= static_cast<size_t>(dimension);
    }
    return result;
}
Value tensor(JNIEnv* env, jobject buffer, const std::vector<int64_t>& dimensions, const OrtMemoryInfo* memory) {
    void* data = env->GetDirectBufferAddress(buffer);
    const auto capacity = env->GetDirectBufferCapacity(buffer);
    const auto count = bytes(dimensions);
    if (!data || capacity < 0 || static_cast<uint64_t>(capacity) != count)
        throw std::runtime_error("UINT16 tensor requires a direct buffer of the exact byte size");
    OrtValue* value = nullptr;
    checked(api().CreateTensorWithDataAsOrtValue(memory, data, count, dimensions.data(), dimensions.size(),
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, &value));
    return Value(value);
}
void fail(JNIEnv* env, const std::exception& error) {
    if (!env->ExceptionCheck()) env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
}
class UtfString {
 public:
    UtfString(JNIEnv* env, jstring text) : env_(env), text_(text), value_(env->GetStringUTFChars(text, nullptr)) {
        if (!value_) throw std::runtime_error("Could not access tensor name");
    }
    ~UtfString() { env_->ReleaseStringUTFChars(text_, value_); }
    const char* get() const { return value_; }
 private:
    JNIEnv* env_;
    jstring text_;
    const char* value_;
};
} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_yy26git_depthanythingv2qnn_UInt16SessionRunner_nativeVersion(JNIEnv* env, jclass) {
    return env->NewStringUTF(OrtGetApiBase()->GetVersionString());
}

extern "C" JNIEXPORT void JNICALL
Java_com_yy26git_depthanythingv2qnn_UInt16SessionRunner_runNative(
        JNIEnv* env, jclass, jobject session_object, jstring input_name, jstring output_name,
        jobject input_buffer, jobject output_buffer, jlongArray input_shape, jlongArray output_shape) {
    try {
        // Private-field ABI is deliberately bounded to the pinned ORT AAR. Java holds a strong
        // reference throughout this synchronous call; the estimator serializes close against Run.
        jclass session_class = env->GetObjectClass(session_object);
        const jfieldID handle_field = env->GetFieldID(session_class, "nativeHandle", "J");
        const jfieldID closed_field = env->GetFieldID(session_class, "closed", "Z");
        if (!handle_field || !closed_field || env->ExceptionCheck()) return;
        const auto handle = env->GetLongField(session_object, handle_field);
        if (!handle || env->GetBooleanField(session_object, closed_field))
            throw std::runtime_error("Cannot run a closed borrowed ORT session");
        auto* session = reinterpret_cast<OrtSession*>(handle);
        OrtMemoryInfo* raw_memory = nullptr;
        checked(api().CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &raw_memory));
        const Memory memory(raw_memory);
        const auto inputs_shape = shape(env, input_shape);
        const auto outputs_shape = shape(env, output_shape);
        if (env->ExceptionCheck()) return;
        const Value input = tensor(env, input_buffer, inputs_shape, memory.get());
        const Value output = tensor(env, output_buffer, outputs_shape, memory.get());
        const UtfString input_text(env, input_name), output_text(env, output_name);
        const char* input_names[] = {input_text.get()};
        const char* output_names[] = {output_text.get()};
        const OrtValue* inputs[] = {input.get()};
        OrtValue* outputs[] = {output.get()};
        checked(api().Run(session, nullptr, input_names, inputs, 1, output_names, 1, outputs));
        // Tensors borrow reusable direct buffers. RAII releases tensor wrappers only, never
        // the Java session, Java storage, provider, allocator, or the factory's cached context.
    } catch (const std::exception& error) { fail(env, error); }
}


// Bulk conversion avoids one managed/direct-buffer boundary crossing per tensor element.
// Float division precedes explicit ties-to-even rounding, independent of the thread's FP mode.
extern "C" JNIEXPORT void JNICALL
Java_com_yy26git_depthanythingv2qnn_UInt16SessionRunner_encodeNative(
        JNIEnv* env, jclass, jfloatArray values, jobject destination, jfloat scale, jint zero_point) {
    try {
        const jsize count = env->GetArrayLength(values);
        auto* output = static_cast<uint16_t*>(env->GetDirectBufferAddress(destination));
        if (!output || env->GetDirectBufferCapacity(destination) != static_cast<jlong>(count) * 2 ||
            !std::isfinite(scale) || scale <= 0 || zero_point < 0 || zero_point > 65535)
            throw std::runtime_error("Invalid bulk UINT16 input boundary");
        std::vector<float> input(count);
        env->GetFloatArrayRegion(values, 0, count, input.data());
        if (env->ExceptionCheck()) return;
        for (jsize i = 0; i < count; ++i) {
            if (!std::isfinite(input[i])) throw std::runtime_error("Depth input contains a non-finite pixel");
            const float quotient = input[i] / scale;
            if (quotient <= -static_cast<float>(zero_point) - 1.0f) { output[i] = 0; continue; }
            if (quotient >= 65536.0f - static_cast<float>(zero_point)) { output[i] = 65535; continue; }
            const double lower = std::floor(static_cast<double>(quotient));
            const double fraction = static_cast<double>(quotient) - lower;
            const int64_t floor_code = static_cast<int64_t>(lower);
            const int64_t rounded = floor_code + ((fraction > 0.5 ||
                (fraction == 0.5 && floor_code % 2 != 0)) ? 1 : 0);
            const int64_t code = rounded + zero_point;
            output[i] = static_cast<uint16_t>(code < 0 ? 0 : code > 65535 ? 65535 : code);
        }
    } catch (const std::exception& error) { fail(env, error); }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_yy26git_depthanythingv2qnn_UInt16SessionRunner_decodeNative(
        JNIEnv* env, jclass, jobject source, jfloat scale, jint zero_point) {
    try {
        const auto capacity = env->GetDirectBufferCapacity(source);
        const auto* input = static_cast<const uint16_t*>(env->GetDirectBufferAddress(source));
        if (!input || capacity < 0 || capacity % 2 != 0 || capacity / 2 > std::numeric_limits<jsize>::max() ||
            !std::isfinite(scale) || scale <= 0 || zero_point < 0 || zero_point > 65535)
            throw std::runtime_error("Invalid bulk UINT16 output boundary");
        const auto count = static_cast<jsize>(capacity / 2);
        std::vector<float> output(count);
        for (jsize i = 0; i < count; ++i) output[i] = (static_cast<int32_t>(input[i]) - zero_point) * scale;
        jfloatArray result = env->NewFloatArray(count);
        if (result) env->SetFloatArrayRegion(result, 0, count, output.data());
        return result;
    } catch (const std::exception& error) { fail(env, error); return nullptr; }
}
