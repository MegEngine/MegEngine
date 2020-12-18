/**
 * \file src/core/include/megbrain/ir/base.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef MGB_BASE
#define MGB_BASE

include "mlir/IR/OpBase.td"

def Mgb_Dialect : Dialect {
  let name = "mgb";
  let cppNamespace = "mgb::dialect";
}

// -- mgb Attr mixin
class MgbAttrWrapperBase<string className> {
  string underlyingType = className;
  int recursionDepth = 0;
}

class MgbHashableAttrMixin {
  string hashFunction = "mgb::hash($0)";
  // return 0 for eq, else for ne
  string cmpFunction = "$0 != $1";
}

class MgbEnumAttrMixin<string namespace, string name, list<string> members> {
  string parentNamespace = namespace;
  string enumName = name;
  list<string> enumMembers = members;
}

class MgbAttrWrapper;
class MgbAliasAttrMixin<Attr base> {
  Attr aliasBase = base;
}

// -- mgb custom Attr
// TODO: CPred and description
class MgbAttrWrapper<string className>:
  Attr<CPred<"true">, "TODO">, MgbAttrWrapperBase<className> {
  let returnType = underlyingType;
}

class HashableAttr<string className>:
    MgbAttrWrapper<className>, MgbHashableAttrMixin;

// -- basic types
class MgbIntegerAttrBase<string CType> : HashableAttr<CType> {
  let storageType = "::mlir::IntegerAttr";
}

class MgbSignlessIntegerAttrBase<string CType> : MgbIntegerAttrBase<CType> {
  let convertFromStorage = "static_cast<" # underlyingType # ">($_self.getInt())";
  let constBuilderCall = "$_builder.getIntegerAttr($_builder.getIntegerType(sizeof(" # underlyingType # ") * 4), $0)";
}

class MgbSignedIntegerAttrBase<string CType> : MgbIntegerAttrBase<CType> {
  let convertFromStorage = "static_cast<" # underlyingType # ">($_self.getSInt())";
  let constBuilderCall = "$_builder.getIntegerAttr($_builder.getIntegerType(sizeof(" # underlyingType # ") * 4, true), $0)";
}

class MgbUnsignedIntegerAttrBase<string CType> : MgbIntegerAttrBase<CType> {
  let convertFromStorage = "static_cast<" # underlyingType # ">($_self.getUInt())";
  let constBuilderCall = "$_builder.getIntegerAttr($_builder.getIntegerType(sizeof(" # underlyingType # ") * 4, false), $0)";
}

def MgbI8Attr: MgbSignlessIntegerAttrBase<"int8_t">;
def MgbI32Attr: MgbSignlessIntegerAttrBase<"int32_t">;
def MgbI64Attr: MgbSignlessIntegerAttrBase<"int64_t">;
def MgbUI32Attr: MgbUnsignedIntegerAttrBase<"uint32_t">;
def MgbUI64Attr: MgbUnsignedIntegerAttrBase<"uint64_t">;
def MgbSizeTAddr: MgbUnsignedIntegerAttrBase<"size_t">;

class MgbFloatAttrBase<string CType, string DType> : HashableAttr<CType> {
  let storageType = "::mlir::FloatAttr";
  let convertFromStorage = "static_cast<" # underlyingType # ">($_self.getValueAsDouble())";
  let constBuilderCall = "$_builder.getFloatAttr($_builder.get" # DType # "Type(), $0)";
}

def MgbF32Attr : MgbFloatAttrBase<"float", "F32">;
def MgbF64Attr : MgbFloatAttrBase<"double", "F64">;

def MgbBoolAttr : HashableAttr<"bool"> {
  let storageType = "::mlir::BoolAttr";
  let constBuilderCall = "$_builder.getBoolAttr($0)";
}

def MgbStringAttr : HashableAttr<"std::string"> {
  let storageType = "::mlir::StringAttr";
  let convertFromStorage = "$_self.getValue().str()";
  let constBuilderCall = "$_builder.getStringAttr($0)"; // llvm::StringRef implicit ctor
}

class MgbArrayAttr<MgbAttrWrapper elem>:
    HashableAttr<"std::vector<" # elem.underlyingType # ">"> {
  let storageType = "::mlir::ArrayAttr";
  let recursionDepth = !add(elem.recursionDepth, 1);
  let convertFromStorage =
    "[&] {\n"
    "    " # underlyingType # " ret" # recursionDepth # ";\n"
    "    std::for_each($_self.begin(), $_self.end(), [&](auto&& i" # recursionDepth # ") {\n"
    "        ret" # recursionDepth # ".push_back(\n"
    "            " # !subst("$_self", "i" # recursionDepth # ".template cast<" # elem.storageType # ">()", "" # elem.convertFromStorage) # "\n"
    "        );\n"
    "    });\n"
    "    return ret" # recursionDepth # ";}()";
  let constBuilderCall =
    "[&] {\n"
    "    std::vector<mlir::Attribute> ret" # recursionDepth # ";\n"
    "    std::for_each($0.begin(), $0.end(), [&](auto&& i" # recursionDepth # ") {\n"
    "        ret" # recursionDepth # ".push_back(\n"
    "            " # !subst("$0", "i" # recursionDepth, "" # elem.constBuilderCall) # "\n"
    "        );\n"
    "    });\n"
    "    return $_builder.getArrayAttr(ret" # recursionDepth # ");"
    "}()";
}

defvar EmptyStrList = !listsplat("", 0);
class StrListAppend<list<string> l, string s> {
  list<string> r = !listconcat(l, !listsplat(s, 1));
}

class TupleConvertFromStorage<MgbAttrWrapper attr, int idx> {
  string r = !subst(
    "$_self",
    "$_self[" # !cast<string>(idx) # "].template cast<"# attr.storageType #">()",
    "" # attr.convertFromStorage);
}

class TupleConstBuilderCall<MgbAttrWrapper attr, int idx> {
  string r = !subst(
    "$0",
    "std::get<" # !cast<string>(idx) # ">($0)",
    "" # attr.constBuilderCall);
}

class ApplyTupleConvertFromStorage<list<MgbAttrWrapper> args> {
  list<string> r = !foldl(
    EmptyStrList, args, l, arg, StrListAppend<l, TupleConvertFromStorage<arg, !size(l)>.r>.r);
}

class ApplyTupleConstBuilderCall<list<MgbAttrWrapper> args> {
  list<string> r = !foldl(
    EmptyStrList, args, l, arg, StrListAppend<l, TupleConstBuilderCall<arg, !size(l)>.r>.r);
}

class MgbTupleAttr<list<MgbAttrWrapper> args>:
    HashableAttr<"std::tuple<" # StrJoin<!foreach(i, args, i.underlyingType)>.result # ">"> {
  let storageType = "::mlir::ArrayAttr";
  let convertFromStorage = "std::make_tuple(" # StrJoin<ApplyTupleConvertFromStorage<args>.r>.result # ")";
  let constBuilderCall = "$_builder.getArrayAttr({" # StrJoin<ApplyTupleConstBuilderCall<args>.r>.result # "})";
}

// -- enum types
class MgbEnumAttr<string namespace, string enumName, list<string> members>:
    HashableAttr<namespace # "::" # enumName>, MgbEnumAttrMixin<namespace, enumName, members> {
  let storageType = "::mlir::IntegerAttr";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getInt())";
  let constBuilderCall = "$_builder.getI32IntegerAttr(static_cast<int32_t>($0))";
  let hashFunction = "mgb::enumhash()($0)";
}

class MgbEnumAliasAttr<string namespace, string enumName, MgbEnumAttr base>:
    MgbEnumAttr<namespace, enumName, base.enumMembers>, MgbAliasAttrMixin<base>;

// -- other types
def MgbDTypeAttr: HashableAttr<"::megdnn::DType"> {
  let storageType = "::mlir::IntegerAttr";
  let convertFromStorage = underlyingType # "::from_enum(static_cast<::megdnn::DTypeEnum>($_self.getInt()))";
  let constBuilderCall = "$_builder.getI32IntegerAttr(static_cast<int32_t>($0.enumv()))";
  let hashFunction = "mgb::hash($0.handle())";
}

def MgbCompNodeAttr: HashableAttr<"::mgb::CompNode"> {
  let storageType = "::mlir::StringAttr";
  let convertFromStorage = underlyingType # "::load($_self.getValue().str())";
  let constBuilderCall = "$_builder.getStringAttr($0.to_string_logical())";
}

def MgbTensorShapeAttr: HashableAttr<"::megdnn::TensorShape"> {
  let storageType = "::mlir::ArrayAttr";
  let hashFunction = "mgb::PODHash<size_t>::perform($0.shape, $0.ndim)";
  let cmpFunction = "!$0.eq_shape($1)";
  defvar elemInst = MgbSizeTAddr;
  let convertFromStorage =
    "[&] {\n"
    "    " # underlyingType # " ret;\n"
    "    std::for_each($_self.begin(), $_self.end(), [&ret](auto&& i) {\n"
    "        ret[ret.ndim ++] = " # !subst("$_self", "i.template cast<"# elemInst.storageType #">()", "" # elemInst.convertFromStorage) # ";\n"
    "    });\n"
    "    return ret;}()";
  let constBuilderCall =
    "[&] {\n"
    "    std::vector<mlir::Attribute> ret;\n"
    "    for (size_t i = 0; i < $0.ndim; ++ i) {\n"
    "        ret.push_back(\n"
    "            " # !subst("$0", "$0[i]", "" # elemInst.constBuilderCall) # "\n"
    "        );\n"
    "    }\n"
    "    return $_builder.getArrayAttr(ret);"
    "}()";
}

class MgbDefaultValuedAttr<MgbAttrWrapper attr, string value>:
    DefaultValuedAttr<attr, value>, MgbAttrWrapperBase<attr.underlyingType> {
  // Note: this class is similar to DefaultValuedAttr but with extra
  // meta informations which are used by mgb dialect tblgen, so this
  // has to be kept up to date with class MgbAttrWrapperMixin
  let recursionDepth = attr.recursionDepth;
}

// -- dnn params
class MgbParamBase<string className> {
  string paramType = className;
  string fullName = "::megdnn::param::" # paramType;
  dag fields = ?;
}

class MgbPackedParamBase<string className, string accessor>:
  MgbParamBase<className> {
  string paramAccessor = accessor;
}

// -- mgb ops
class MgbHashableOpMixin {
  string hashFunction = ?;
  string cmpFunction = ?;
}

class MgbOp<string mnemonic, list<MgbParamBase> params=[], list<OpTrait> traits=[]>:
    Op<Mgb_Dialect, mnemonic, traits> {
  dag inputs = (ins);
  dag extraArguments = (ins);
  // TODO: remove it
  code extraOpdefDecl = ?;

  let arguments = !con(
    !foldl(inputs, params, args, param, !con(args, param.fields)),
    extraArguments);

  list<MgbParamBase> dnnParams = params;
}

class MgbHashableOp<string mnemonic, list<MgbParamBase> params=[], list<OpTrait> traits=[]>:
  MgbOp<mnemonic, params, traits>, MgbHashableOpMixin;

#endif // MGB_BASE
