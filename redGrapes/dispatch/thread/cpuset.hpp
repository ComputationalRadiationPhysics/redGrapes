/* Copyright 2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace redGrapes
{
    namespace dispatch
    {
        namespace thread
        {

            void pin_cpu(unsigned);
            void unpin_cpu();

        } // namespace thread
    } // namespace dispatch
} // namespace redGrapes
