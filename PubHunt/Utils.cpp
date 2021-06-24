#include "Utils.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <string>
#include <string.h>

//using namespace std;

std::vector<uint8_t> hex2bytes(const std::string& s)
{
	constexpr size_t width = sizeof(uint8_t) * 2;
	std::vector<uint8_t> v;
	v.reserve((s.size() + width - 1) / width);
	for (auto it = s.crbegin(); it < s.crend(); it += width)
	{
		auto begin = std::min(s.crend(), it + width).base();
		auto end = it.base();
		std::string slice(begin, end);
		uint8_t value = std::stoul(slice, 0, 16);
		v.push_back(value);
	}
	std::reverse(v.begin(), v.end());
	return v;
}

int getInt(std::string name, const char* v) {

	int r;

	try {

		r = std::stoi(std::string(v));

	}
	catch (std::invalid_argument&) {

		printf("Invalid %s argument, number expected\n", name.c_str());
		exit(-1);

	}

	return r;

}

void getInts(std::string name, std::vector<int>& tokens, const std::string& text, char sep)
{
	size_t start = 0, end = 0;
	tokens.clear();
	int item;

	try {

		while ((end = text.find(sep, start)) != std::string::npos) {
			item = std::stoi(text.substr(start, end - start));
			tokens.push_back(item);
			start = end + 1;
		}

		item = std::stoi(text.substr(start));
		tokens.push_back(item);

	}
	catch (std::invalid_argument&) {

		printf("Invalid %s argument, number expected\n", name.c_str());
		exit(-1);

	}
}

void parseFile(std::string fileName, std::vector<std::vector<uint8_t>>& inputHashes)
{
	inputHashes.clear();
	// Check file
	FILE* fp = fopen(fileName.c_str(), "rb");
	if (fp == NULL) {
		::printf("Error: Cannot open %s %s\n", fileName.c_str(), strerror(errno));
	}
	else {
		fclose(fp);
		int nbLine = 0;
		std::string line;
		std::ifstream inFile(fileName);
		while (getline(inFile, line)) {
			// Remove ending \r\n
			int l = (int)line.length() - 1;
			while (l >= 0 && isspace(line.at(l))) {
				line.pop_back();
				l--;
			}
			if (line.length() > 0) {
				auto ret = hex2bytes(line);
				if (ret.size() == 20) {
					inputHashes.push_back(ret);
				}
				else {
					::printf("Error: Cannot read hash at line %d, \n", nbLine);
				}
			}
			nbLine++;
		}
		//::printf("Loaded range history: %d\n", nbLine);
	}
}