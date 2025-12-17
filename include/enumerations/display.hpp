#pragma once

namespace specfem {
namespace display {

enum class format { PNG, JPG, on_screen, vtkhdf };

enum class component { x, y, z, magnitude };

std::string to_string(const format &fmt);
std::string to_string(const component &comp);

} // namespace display
} // namespace specfem
