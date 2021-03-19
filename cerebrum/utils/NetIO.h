//
// Created by Nik Willwerth on 3/19/21.
//

#ifndef CEREBRUM_NETIO_H
#define CEREBRUM_NETIO_H


#include <string>

class NetIO {
public:
    static void downloadFile(const std::string& url, const std::string& outputFile);
};


#endif //CEREBRUM_NETIO_H
