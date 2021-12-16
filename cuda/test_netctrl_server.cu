#include "net_control.hpp"

int main() {
	printf("Waiting...\n");
	SignalContext ctx = initializeSignalListener(7000);
	waitForSignal(ctx);
	destroySignalContext(ctx);

	return 0;
}