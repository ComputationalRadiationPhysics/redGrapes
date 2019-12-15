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
    redGrapes::FieldResource< std::vector<int> > a;
    redGrapes::IOResource<int> b;
    redGrapes::IOResource<int> c;

    redGrapes::ResourceUser user1({
        a.read(), // complete resource
        a.write().area( {0}, {10} ), // write only indices 0 to 10
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

