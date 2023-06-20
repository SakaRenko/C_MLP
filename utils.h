#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED


#include <string>
#include <ctime>

std::string get_time();

void enable_debug_mode();

void print(const std::string &msg, const std::string &component);
void print_info(const std::string &msg, const std::string &component);
void print_debug(const std::string &msg, const std::string &component);
void print_warn(const std::string &msg, const std::string &component);
void print_error(const std::string &msg, const std::string &component);
void print_fatal(const std::string &msg, const std::string &component);


#endif // UTILS_H_INCLUDED
