#ifdef _WIN32

#pragma comment(lib, "Ws2_32.lib")
#include "win_udp.hpp"

#elif __linux__

#include "linux_udp.hpp"

#endif

struct SignalContext {
#ifdef _WIN32
	WSASession *session;
	WinUDPSocket *socket;
#endif
};

SignalContext initializeClient() {
	#ifdef _WIN32

	WSASession *session = new WSASession;
	WinUDPSocket *socket = new WinUDPSocket;
	return SignalContext{session, socket};

	#endif
}

SignalContext initializeSignalListener(
	unsigned int port,
	unsigned long timeout = 0
) {
	#ifdef _WIN32

	SignalContext ctx = initializeClient();

	ctx.socket->Bind(port);
	if (timeout > 0) { ctx.socket->SetReadTimeout(timeout); }
	return SignalContext{ctx.session, ctx.socket};

	#endif
}

void destroySignalContext(SignalContext ctx) {
	#ifdef _WIN32
	ctx.socket->close();
	ctx.session->close();
	delete ctx.session;
	delete ctx.socket;
	#endif
}

bool waitForSignal(SignalContext context) {
	#ifdef _WIN32
		try {
			char *buff = (char*)malloc(sizeof(char) * 1024);
			context.socket->RecvFrom(buff, sizeof(char) * 1024, 0);
			free(buff);
		} catch (RecvTimeout &e) {
			return false;
		}

		return true;
	#endif
}

void sendSignal(SignalContext context, std::string IP, unsigned short PORT) {
	#ifdef _WIN32
		char a = 65;
		context.socket->SendTo(IP, PORT, &a, 1);
	#endif
}
