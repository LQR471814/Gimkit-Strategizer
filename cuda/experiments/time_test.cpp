#include <iostream>
#include <thread>
#include <chrono>

int main()
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	std::chrono::system_clock::time_point timePt =
		std::chrono::system_clock::now() + std::chrono::seconds(5);
	std::this_thread::sleep_until(timePt);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout
		<< "Time difference = "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(
			end - begin
		).count()
		<< std::endl;

	return 0;
}