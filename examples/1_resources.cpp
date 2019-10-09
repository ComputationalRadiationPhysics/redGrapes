/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <rmngr/resource/ioresource.hpp>
#include <rmngr/resource/fieldresource.hpp>
#include <rmngr/resource/resource_user.hpp>

int main(int, char*[])
{
    rmngr::FieldResource<2> a;
    rmngr::IOResource b;
    rmngr::IOResource c;

    rmngr::ResourceUser user1({
        a.read(), // complete resource
        a.write({{
            {0, 20}, // area of 1st dimension
            {0, 10}  // area of 2nd dimension
        }}),
        b.write()
    });

    rmngr::ResourceUser user2({
        b.read()
    });

    rmngr::ResourceUser user3({
        b.read(),
        c.write()
    });

    std::cout << "is_serial(user1,user1) = " << rmngr::ResourceUser::is_serial(user1,user1) << std::endl;
    std::cout << "is_serial(user1,user2) = " << rmngr::ResourceUser::is_serial(user1,user2) << std::endl;
    std::cout << "is_serial(user1,user3) = " << rmngr::ResourceUser::is_serial(user1,user3) << std::endl;
    std::cout << "is_serial(user2,user3) = " << rmngr::ResourceUser::is_serial(user2,user3) << std::endl;

    return 0;
} done
