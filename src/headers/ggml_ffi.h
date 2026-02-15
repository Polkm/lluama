// GGML Backend API
// Minimal FFI definitions for backend loading functions

bool ggml_backend_load_all(void);
bool ggml_backend_load(const char * path);
