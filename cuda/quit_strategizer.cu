#include "CLI11.hpp"
#include "net_control.hpp"

int main(int argc, char** argv) {
	CLI::App app{"A convenience utility for quitting the computation program."};

	unsigned int port = 7000;
	app.add_option("-p,--port", port, "The port that the program is listening on");

	CLI11_PARSE(app, argc, argv);

	SignalContext ctx = initializeClient();
	sendSignal(ctx, "127.0.0.1", port);
	destroySignalContext(ctx);

	printf("Signal sent\n");

	return 0;
}