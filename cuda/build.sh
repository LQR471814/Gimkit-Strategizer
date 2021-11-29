if test -f "CLI11.hpp"; then
	wget https://github.com/CLIUtils/CLI11/releases/download/v2.1.2/CLI11.hpp
fi
nvcc strategizer.cu --use_fast_math