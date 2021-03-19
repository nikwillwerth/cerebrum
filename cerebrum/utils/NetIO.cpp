//
// Created by Nik Willwerth on 3/19/21.
//

#include "NetIO.h"
#include <curl/curl.h>
#include <iostream>

void NetIO::downloadFile(const std::string& url, const std::string& outputFile) {
    std::cout << "Downloading " << url << "..." << std::endl;

    FILE *fp;
    CURL *curl = curl_easy_init();

    if(curl)
    {
        fp = fopen(outputFile.c_str(),"wb");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

        curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        fclose(fp);
    }
}
