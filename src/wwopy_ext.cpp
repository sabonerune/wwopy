/*
SPDX-FileCopyrightText: (c) 2024, sabonerune
SPDX-License-Identifier: BSD-2-Clause
*/

#include "wwopy_init.hpp"

// NOLINTNEXTLINE
NB_MODULE(wwopy_ext, m) {
  cheeptrick_init(m);
  d4c_init(m);
  dio_init(m);
  harvest_init(m);
  stonemask_init(m);
  synthesis_init(m);
}
