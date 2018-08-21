#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <list>
#include "FileInput.h"

using namespace std;

void countKs(char *abcdFile, int &count1, int &count3) {
	string line;

	ifstream myfile(abcdFile);
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			count3++;

			string::size_type n = line.find(" ");
			string token = line.substr(0, n);
			//cout << token << endl;

			//If tokens[0] == "ABCD" then this defines a new k, otherwise it is a number
			if (token.compare("ABCD") == 0) {
				count1++;
				//cout << "We're here!" << endl;
			}
		}
		myfile.close();
	}

	else cout << "Unable to open file first time" << endl;
}
