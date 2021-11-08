#include "logger.hpp"

int main() {
	int roots = 1000;
	LogContext ctx = logGrid(roots, 100);

	for (int i = 0; i < roots; i++) {
		modGrid(ctx, i);
	};

	gridFinish(ctx);
	printf("Done!\n");

	return 0;
}