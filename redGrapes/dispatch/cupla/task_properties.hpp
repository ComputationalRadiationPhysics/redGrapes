/* Copyright 2020 Michael Sippel
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
        namespace cupla
        {

            struct CuplaTaskProperties
            {
                std::optional<cuplaEvent_t> cupla_event;

                CuplaTaskProperties()
                {
                }

                template<typename PropertiesBuilder>
                struct Builder
                {
                    PropertiesBuilder& builder;

                    Builder(PropertiesBuilder& b) : builder(b)
                    {
                    }
                };

                struct Patch
                {
                    template<typename PatchBuilder>
                    struct Builder
                    {
                        Builder(PatchBuilder&)
                        {
                        }
                    };
                };

                void apply_patch(Patch const&){};
            };

        } // namespace cupla
    } // namespace dispatch
} // namespace redGrapes
