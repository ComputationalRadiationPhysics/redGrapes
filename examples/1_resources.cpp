/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/resource/fieldresource.hpp>
#include <redGrapes/resource/resource_user.hpp>

int main(int, char*[])
{
    redGrapes::FieldResource<2> a;
    redGrapes::IOResource b;
    redGrapes::IOResource c;

    redGrapes::ResourceUser user1({
        a.read(), // complete resource
        a.write({{
            {0, 20}, // area of 1st dimension
            {0, 10}  // area of 2nd dimension
        }}),
        b.write()
    });

    redGrapes::ResourceUser user2({
        b.read()
    });

    redGrapes::ResourceUser user3({
        b.read(),
        c.write()
    });

    std::cout << "is_serial(user1,user1) = " << redGrapes::ResourceUser::is_serial(user1,user1) << std::endl;
    std::cout << "is_serial(user1,user2) = " << redGrapes::ResourceUser::is_serial(user1,user2) << std::endl;
    std::cout << "is_serial(user1,user3) = " << redGrapes::ResourceUser::is_serial(user1,user3) << std::endl;
    std::cout << "is_serial(user2,user3) = " << redGrapes::ResourceUser::is_serial(user2,user3) << std::endl;

    return 0;
}

