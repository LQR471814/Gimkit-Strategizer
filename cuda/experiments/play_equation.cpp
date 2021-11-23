#include <math.h>
#include <iostream>

struct GoalResult {
	int problems;
	double money;
};

struct GoalResult playGoal(double init, float goal) {
	float mq = 1;
	float sb = 2;
	float mu = 1;

	float a = mu*sb;
	float b = -mu*(sb - 2*mq);
	float c = 2*(init-goal);

	float problems = ceilf(
		(-b + sqrtf(pow(b, 2) - 4*a*c)) / (2*a)
	);

	double money = init + (
		mu*problems * (
			2*mq + sb*(problems - 1)
		)
	) / 2;

	return GoalResult{
		int(problems),
		money,
	};
}

int main() {
	GoalResult result = playGoal(0, 100.0);

	printf("Problems %d Money %f", result.problems, result.money);
}
