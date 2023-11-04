#include <c10/core/Curve.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <exception>
#include <string>
#include <vector>

namespace c10 {

std::string CurveFamilyName(CurveFamily d, bool lower_case) {
  switch (d) {
    // I considered instead using ctype::tolower to lower-case the strings
    // on the fly, but this seemed a bit much.
    case CurveFamily::BN254:
      return lower_case ? "bn254" : "BN254";
    case CurveFamily::BLS12_377:
      return lower_case ? "bls12-377" : "BLS12-377";
    case CurveFamily::BLS12_381:
      return lower_case ? "bls12-381" : "BLS12-381";
    case CurveFamily::MNT4753:
      return lower_case ? "mnt4753" : "MNT4753";
    default:
      TORCH_CHECK(
          false,
          "Unknown elliptic curve: ",
          static_cast<int16_t>(d),
          ". If you have recently updated the caffe2.proto file to add a new "
          "curve family, did you forget to update the CurveFamilyName() "
          "function to reflect such recent changes?");
      // The below code won't run but is needed to suppress some compiler
      // warnings.
      return "";
  }
}

std::string CurveGroupName(CurveGroup d, bool lower_case) {
  switch (d) {
    // I considered instead using ctype::tolower to lower-case the strings
    // on the fly, but this seemed a bit much.
    case CurveGroup::G1:
      return lower_case ? "g1" : "G1";
    case CurveGroup::G2:
      return lower_case ? "g2" : "G2";
    default:
      TORCH_CHECK(
          false,
          "Unknown elliptic curve group: ",
          static_cast<int16_t>(d),
          ". If you have recently updated the caffe2.proto file to add a new "
          "curve group, did you forget to update the CurveGroupName() "
          "function to reflect such recent changes?");
      // The below code won't run but is needed to suppress some compiler
      // warnings.
      return "";
  }
}

std::string CurveFieldName(CurveField d, bool lower_case) {
  switch (d) {
    // I considered instead using ctype::tolower to lower-case the strings
    // on the fly, but this seemed a bit much.
    case CurveField::Fr:
      return lower_case ? "fr" : "Fr";
    case CurveField::Fq:
      return lower_case ? "fq" : "Fq";
    default:
      TORCH_CHECK(
          false,
          "Unknown elliptic curve field: ",
          static_cast<int16_t>(d),
          ". If you have recently updated the caffe2.proto file to add a new "
          "curve field, did you forget to update the CurveFieldName() "
          "function to reflect such recent changes?");
      // The below code won't run but is needed to suppress some compiler
      // warnings.
      return "";
  }
}


namespace {
CurveFamily parse_curve_family_type(const std::string& curve_family_string) {
  static const std::array<
      std::pair<const char*, CurveFamily>,
      static_cast<size_t>(CurveFamily::COMPILE_TIME_MAX_CURVE_FAMILY_TYPES)>
      types = {{
          {"bn254", CurveFamily::BN254},
          {"bls12-377", CurveFamily::BLS12_377},
          {"bls12-381", CurveFamily::BLS12_381},
          {"mnt4753", CurveFamily::MNT4753},
      }};
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [&curve_family_string](const std::pair<const char*, CurveFamily>& p) {
        return p.first && p.first == curve_family_string;
      });
  if (device != types.end()) {
    return device->second;
  }
  std::vector<const char*> family_names;
  for (const auto& it : types) {
    if (it.first) {
      family_names.push_back(it.first);
    }
  }
  TORCH_CHECK(
      false,
      "Expected one of ",
      c10::Join(", ", family_names),
      " device type at start of device string: ",
      curve_family_string);
}

CurveGroup parse_curve_group_type(const std::string& curve_group_string) {
  static const std::array<
      std::pair<const char*, CurveGroup>,
      static_cast<size_t>(CurveGroup::COMPILE_TIME_MAX_CURVE_GROUP_TYPES)>
      types = {{
          {"g1", CurveGroup::G1},
          {"g2", CurveGroup::G2},
      }};
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [&curve_group_string](const std::pair<const char*, CurveGroup>& p) {
        return p.first && p.first == curve_group_string;
      });
  if (device != types.end()) {
    return device->second;
  }
  std::vector<const char*> group_names;
  for (const auto& it : types) {
    if (it.first) {
      group_names.push_back(it.first);
    }
  }
  TORCH_CHECK(
      false,
      "Expected one of ",
      c10::Join(", ", group_names),
      " device type at start of device string: ",
      curve_group_string);
}

CurveField parse_curve_field_type(const std::string& curve_field_string) {
  static const std::array<
      std::pair<const char*, CurveField>,
      static_cast<size_t>(CurveField::COMPILE_TIME_MAX_CURVE_FIELD_TYPES)>
      types = {{
          {"fr", CurveField::Fr},
          {"fq", CurveField::Fq},
      }};
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [&curve_field_string](const std::pair<const char*, CurveField>& p) {
        return p.first && p.first == curve_field_string;
      });
  if (device != types.end()) {
    return device->second;
  }
  std::vector<const char*> field_names;
  for (const auto& it : types) {
    if (it.first) {
      field_names.push_back(it.first);
    }
  }
  TORCH_CHECK(
      false,
      "Expected one of ",
      c10::Join(", ", field_names),
      " device type at start of device string: ",
      curve_field_string);
}

} // namespace

Curve::Curve(const std::string& curve_family_string, const std::string& curve_group_string, const std::string& curve_field_string) {
  TORCH_CHECK(!curve_family_string.empty(), "Curve family string must not be empty");
  curve_family_ = parse_curve_family_type(curve_family_string);
  TORCH_CHECK(!curve_group_string.empty(), "Curve group string must not be empty");
  curve_group_ = parse_curve_group_type(curve_group_string);
  TORCH_CHECK(!curve_field_string.empty(), "Curve field string must not be empty");
  curve_field_ = parse_curve_field_type(curve_field_string);
}

std::string Curve::str() const {
  std::string str = "<";
  str += CurveFamilyName(curve_family(), /*lower_case=*/false);
  str += ", ";
  str += CurveGroupName(curve_group(), /*lower_case=*/false);
  str += ", ";
  str += CurveFieldName(curve_field(), /*lower_case=*/false);
  str += ">";
  return str;
}

std::ostream& operator<<(std::ostream& stream, const Curve& device) {
  stream << device.str();
  return stream;
}

} // namespace c10
