//
// Created by jason on 2025/7/27.
//
#include "framework_common.h"
#include "framework_model.h"

struct llama_file {

#if defined(_WIN32)
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    HANDLE fp_win32;
    size_t size;

private:
    std::string GetErrorMessageWin32(DWORD error_code) const {
        std::string ret;
        LPSTR lpMsgBuf = NULL;
        DWORD bufLen = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                    NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&lpMsgBuf, 0, NULL);
        if (!bufLen) {
            ret = format("Win32 error code: %lx", error_code);
        } else {
            ret = lpMsgBuf;
            LocalFree(lpMsgBuf);
        }

        return ret;
    }

public:

    llama_file(const char * fname, const char * mode) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        // SetFilePointerEx returns the current position when seeking relative 0 bytes
        LARGE_INTEGER li;
        li.QuadPart = 0;
        BOOL ret = SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }

        return li.QuadPart;
    }

    void seek(size_t offset, int whence) const {
        // no need to convert SEEK_* to FILE_*. The enums are the same.
        // Still, keep static asserts to avoid failures in the future.
        static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
        static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
        static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");

        LARGE_INTEGER li;
        li.QuadPart = offset;
        BOOL ret = SetFilePointerEx(fp_win32, li, NULL, whence);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }
    }

    void read_raw(void * ptr, size_t len) const {
        // On Win32 ReadFile is significant faster than fread which is again significant faster than std::fstream. Thus
        // use the Win32 API to do file io instead of the C/C++ library functions.

        // There are conditions under which ReadFile cannot read chunks >64MB.
        // Thus split the operation into smaller chunks if len exceeds this limit.
        size_t bytes_read = 0;
        while (bytes_read < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_read, 64*1024*1024);
            DWORD chunk_read = 0;
            BOOL result = ReadFile(fp_win32, reinterpret_cast<char*>(ptr) + bytes_read, chunk_size, &chunk_read, NULL);
            if (!result) {
                throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_read < chunk_size || chunk_read == 0) {
                throw std::runtime_error("unexpectedly reached end of file");
            }

            bytes_read += chunk_read;
        } ;
    }

    uint32_t read_u32() const {
        uint32_t val;
        read_raw(&val, sizeof(val));
        return val;
    }

    void write_raw(const void * ptr, size_t len) const {
        // There are conditions under which WriteFile cannot write chunks >64MB.
        // Thus split the operation into smaller chunks if len exceeds this limit.
        size_t bytes_written = 0;
        while (bytes_written < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_written, 64*1024*1024);
            DWORD chunk_written = 0;
            BOOL result = WriteFile(fp_win32, reinterpret_cast<char const*>(ptr) + bytes_written, chunk_size, &chunk_written, NULL);
            if (!result) {
                throw std::runtime_error(format("write error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_written < chunk_size || chunk_written == 0) {
                throw std::runtime_error("unexpectedly failed to write bytes");
            }

            bytes_written += chunk_written;
        }
    }

    void write_u32(std::uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
#else
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    size_t size;

    llama_file(const char * fname, const char * mode) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        if (ret == -1) {
            throw std::runtime_error(format("ftell error: %s", strerror(errno)));
        }

        return (size_t) ret;
    }

    void seek(size_t offset, int whence) const {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        if (ret != 0) {
            throw std::runtime_error(format("seek error: %s", strerror(errno)));
        }
    }

    void read_raw(void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error("unexpectedly reached end of file");
        }
    }

    uint32_t read_u32() const {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(std::uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
#endif
};

namespace ggml_runtime
{
    GGUFLoader::GGUFLoader(const std::string& path)
    {
        struct ggml_context* ctx = NULL;
        struct gguf_init_params params = {
           true,
            &ctx,
        };
        m_context.reset(gguf_init_from_file(path.c_str(), params));
        if (!m_context) {
            throw std::runtime_error("Failed to load GGML file: " + path);
        }

        m_file = new llama_file(path.c_str(), "rb");
        GGMLF_LOG_INFO("GGUF file size: %ld\n", m_file->size);

        // list tensors
        auto n_tensors = gguf_get_n_tensors(m_context.get());
        GGMLF_LOG_INFO("GGUF has %d tensors\n", n_tensors);

        auto data_offset = gguf_get_data_offset(m_context.get());
        // it seems the offset is ascending order, so we can calculate the approximate tensor size
        auto max_tensor_size = 0;
        auto last_tensor_offset = 0;
        auto offset_diff = 0;
        for (int i = 0; i < n_tensors; i++) {
            auto tensor_name = gguf_get_tensor_name(m_context.get(), i);
            auto tensor_type = gguf_get_tensor_type(m_context.get(), i);
            auto tensor_offset = gguf_get_tensor_offset(m_context.get(), i) + data_offset;
            auto tensor_info = std::tuple<ggml_type, uint64_t>(tensor_type, tensor_offset);
            m_tensor_infos.emplace(tensor_name, tensor_info);
            if (last_tensor_offset != 0)
            {
                offset_diff = tensor_offset - last_tensor_offset;
                if (offset_diff > max_tensor_size)
                {
                    max_tensor_size = offset_diff;
                }
            }
            last_tensor_offset = tensor_offset;
        }
        offset_diff = m_file->size - last_tensor_offset;
        if (offset_diff > max_tensor_size)
        {
            max_tensor_size = offset_diff;
        }
        auto tensor_size_mb = max_tensor_size / 1024 / 1024;
        GGMLF_LOG_INFO("Max tensor size: %d MB\n", tensor_size_mb);
        m_tensor_buffer.resize((tensor_size_mb + 1) * 1024 * 1024);
    }

    const char* GGUFLoader::get_tensor_file_data(const std::string& tensor_name, size_t size)
    {
        auto it = m_tensor_infos.find(tensor_name);
        if (it == m_tensor_infos.end()) {
            throw std::runtime_error("Tensor not found: " + tensor_name);
        }
        auto tensor_info = it->second;
        auto tensor_offset = std::get<1>(tensor_info);

        if (tensor_offset + size > m_file->size) {
            throw std::runtime_error("Tensor data out of range: " + tensor_name);
        }
        // seek to the tensor offset and read the tensor data
        m_file->seek(tensor_offset, SEEK_SET);
        m_file->read_raw(m_tensor_buffer.data(), size);
        return m_tensor_buffer.data();
    }

    ggml_type GGUFLoader::get_tensor_type(const std::string& tensor_name)
    {
        auto it = m_tensor_infos.find(tensor_name);
        if (it == m_tensor_infos.end()) {
            throw std::runtime_error("Tensor not found: " + tensor_name);
        }
        auto tensor_info = it->second;
        return std::get<0>(tensor_info);
    }

}