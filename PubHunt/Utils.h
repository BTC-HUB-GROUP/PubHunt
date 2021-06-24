#pragma once
#include <string>
#include <vector>

std::vector<uint8_t> hex2bytes(const std::string& s);

int getInt(std::string name, const char* v);

void getInts(std::string name, std::vector<int>& tokens, const std::string& text, char sep);

void parseFile(std::string fileName, std::vector<std::vector<uint8_t>>& inputHashes);

