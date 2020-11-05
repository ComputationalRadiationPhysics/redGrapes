#<<<<>>>><<>><><<>><<<>>><<>><><<>><<<<>>>><<>><><<>><<<>>><<>><><<>><<<<>>>>#
# Copyright 2020 Michael Sippel                                              #
#                                                                            #
# This Source Code Form is subject to the terms of the Mozilla Public        #
# License, v. 2.0. If a copy of the MPL was not distributed with this        #
# file, You can obtain one at http://mozilla.org/MPL/2.0/.                   #
#<<<<>>>><<>><><<>><<<>>><<>><><<>><<<<>>>><<>><><<>><<<>>><<>><><<>><<<<>>>>#

#!/bin/sh

prettify_log() {
    gawk \
    '{ \
        if(match($0, /(.*emplace_task )(.*)/, a) != 0) \
        { \
             gsub(/"/, "\\\"", a[2]); \
             print a[1]; \
             system("echo \""a[2]"\" | jq -C"); \
        } \
        else { print } \
     }'
}

task_json() {
    grep -Po --color=never "emplace_task \K.*$"
}


