if not exist CLI11.hpp (
	certutil -urlcache -split -f https://github.com/CLIUtils/CLI11/releases/download/v2.1.2/CLI11.hpp CLI11.hpp
)
nvcc strategizer.cu