#include <windows.h>
#include <iostream>

struct LogContext {
	int consoleWidth;
	int rows;
};

LogContext logGrid(int roots, int width) {
	if (width < 0) {
		CONSOLE_SCREEN_BUFFER_INFO csbi;
		GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);

		width = csbi.srWindow.Right - csbi.srWindow.Left;
	};

	std::string s = "";
	int charactersAppended = 0;
	int rows = 0;
	for (int i = 0; i < roots; i++) {
		s.append("·");
		charactersAppended++;
		if (charactersAppended == width) {
			s.append("\n");
			rows++;
			charactersAppended = 0;
		};
	};

	if (charactersAppended != 0) {
		s.append("\n");
		rows++;
	};

	std::cout << s;
	printf("%c[%dA\r", 27, rows); //? reset cursor to start of grid
	return LogContext{width, rows};
}

void modGrid(LogContext ctx, int index) {
	int y = index / ctx.consoleWidth;
	int x = index % ctx.consoleWidth;

	//? go to position
	if (y != 0)
		printf("%c[%dB", 27, y);
	if (x != 0)
		printf("%c[%dC", 27, x);

	printf("█");
	if (y != 0)
		printf("%c[%dA", 27, y); //? return to start

	printf("\r");
}

void gridFinish(LogContext ctx) {
	printf("%c[%dB%c[%dC\n", 27, ctx.rows-1, 27, ctx.consoleWidth);
}
