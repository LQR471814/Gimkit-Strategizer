#include <iostream>
#include <cstring>

#define BUFSIZE 512
#define PACKETSIZE sizeof(MSG)

using namespace std;

typedef struct MSG
{
	int type;
	int priority;
	int sender;
	char message[BUFSIZE];
} MSG;

void serialize(MSG *msgPacket, char *data);
void deserialize(char *data, MSG *msgPacket);
void printMsg(MSG *msgPacket);

int main()
{
	MSG *newMsg = new MSG;
	newMsg->type = 1;
	newMsg->priority = 9;
	newMsg->sender = 2;
	strcpy(newMsg->message, "hello from server\0");
	printMsg(newMsg);

	char data[PACKETSIZE];

	serialize(newMsg, data);

	MSG *temp = new MSG;
	deserialize(data, temp);
	printMsg(temp);

	return 0;
}

void serialize(MSG *msgPacket, char *data)
{
	int *q = (int *)data;
	*q = msgPacket->type;
	q++;
	*q = msgPacket->priority;
	q++;
	*q = msgPacket->sender;
	q++;

	char *p = (char *)q;
	int i = 0;
	while (i < BUFSIZE)
	{
		*p = msgPacket->message[i];
		p++;
		i++;
	}
}

void deserialize(char *data, MSG *msgPacket)
{
	int *q = (int *)data;
	msgPacket->type = *q;
	q++;
	msgPacket->priority = *q;
	q++;
	msgPacket->sender = *q;
	q++;

	char *p = (char *)q;
	int i = 0;
	while (i < BUFSIZE)
	{
		msgPacket->message[i] = *p;
		p++;
		i++;
	}
}

void printMsg(MSG *msgPacket)
{
	cout << msgPacket->type << endl;
	cout << msgPacket->priority << endl;
	cout << msgPacket->sender << endl;
	cout << msgPacket->message << endl;
}