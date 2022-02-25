#include <stdio.h>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/decryption/rc4/rc4_cryption_base.h"
#include "../src/decryption/rc4_cryption.h"

using namespace lite;

std::shared_ptr<void> read_file(std::string file_path, size_t& size) {
    FILE* fin = fopen(file_path.c_str(), "rb");
    if (!fin) {
        printf("failed to open %s.", file_path.c_str());
    };
    fseek(fin, 0, SEEK_END);
    size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    void* ptr = malloc(size);
    std::shared_ptr<void> buf{ptr, ::free};
    fread(buf.get(), 1, size, fin);
    fclose(fin);
    return buf;
}

void write_file(std::string file_path, const std::vector<uint8_t>& data) {
    FILE* fin = fopen(file_path.c_str(), "wb");
    if (!fin) {
        printf("failed to open %s.", file_path.c_str());
    };
    fwrite(data.data(), 1, data.size(), fin);
    fclose(fin);
}

typedef int (*CommandHandler)(int, char**);

const char* usage =
        "Usage:\n"
        " rc4_encryptor encrypt_predefined_rc4 <input file> <output file>\n"
        " rc4_encryptor encrypt_rc4 <hash key> <enc key> <input file> <output "
        "file>\n"
        " rc4_encryptor encrypt_predefined_sfrc4 <input file> <output file>\n"
        " rc4_encryptor encrypt_sfrc4 <hash key> <enc key> <input file> "
        "<output "
        "file>\n"
        " rc4_encryptor hash <input file>\n";

int command_encrypt_predefined_rc4(int argc, char** argv) {
    if (argc != 4) {
        printf("Invalid encrypt_predefined_rc4 arguments.\n");
        return 1;
    }

    const char* input_file_path = argv[2];
    const char* output_file_path = argv[3];

    size_t size = 0;
    auto keys = RC4::get_decrypt_key();
    auto input = read_file(input_file_path, size);
    printf("Reading input file ...\n");
    auto output = RC4::encrypt_model(input.get(), size, keys);

    write_file(output_file_path, output);

    printf("Done.\n");
    return 0;
}

int command_encrypt_rc4(int argc, char** argv) {
    if (argc != 6) {
        printf("Invalid encrypt_rc4 arguments.\n");
        return 1;
    }

    uint64_t hash_key = std::stoull(argv[2], 0, 0);
    uint64_t enc_key = std::stoull(argv[3], 0, 0);
    const char* input_file_path = argv[4];
    const char* output_file_path = argv[5];

    std::vector<uint8_t> keys(128, 0);
    uint64_t* data = reinterpret_cast<uint64_t*>(keys.data());
    data[0] = hash_key;
    data[1] = enc_key;

    size_t size = 0;
    auto input = read_file(input_file_path, size);
    printf("Reading input file ...\n");
    auto output = RC4::encrypt_model(input.get(), size, keys);

    printf("Encrypting ...\n");
    write_file(output_file_path, output);

    printf("Done.\n");
    return 0;
}

int command_encrypt_predefined_sfrc4(int argc, char** argv) {
    if (argc != 4) {
        printf("Invalid encrypt_predefined_rc4 arguments.\n");
        return 1;
    }

    const char* input_file_path = argv[2];
    const char* output_file_path = argv[3];

    size_t size = 0;
    auto keys = SimpleFastRC4::get_decrypt_key();
    auto input = read_file(input_file_path, size);
    printf("Reading input file ...\n");
    auto output = SimpleFastRC4::encrypt_model(input.get(), size, keys);

    write_file(output_file_path, output);

    printf("Done.\n");
    return 0;
}

int command_encrypt_sfrc4(int argc, char** argv) {
    if (argc != 6) {
        printf("Invalid encrypt_rc4 arguments.\n");
        return 1;
    }

    uint64_t hash_key = std::stoull(argv[2], 0, 0);
    uint64_t enc_key = std::stoull(argv[3], 0, 0);
    const char* input_file_path = argv[4];
    const char* output_file_path = argv[5];

    std::vector<uint8_t> keys(128, 0);
    uint64_t* data = reinterpret_cast<uint64_t*>(keys.data());
    data[0] = hash_key;
    data[1] = enc_key;

    size_t size = 0;
    auto input = read_file(input_file_path, size);
    printf("Reading input file ...\n");
    auto output = SimpleFastRC4::encrypt_model(input.get(), size, keys);

    printf("Encrypting ...\n");
    write_file(output_file_path, output);

    printf("Done.\n");
    return 0;
}

int command_hash(int argc, char** argv) {
    if (argc != 3) {
        printf("Invalid hash arguments.\n");
        return 1;
    }

    const char* input_file_path = argv[2];

    size_t len = 0;
    auto input = read_file(input_file_path, len);

    rc4::FastHash64 hasher(rc4::key_gen_hash_key());
    auto start = static_cast<const char*>(input.get());

    auto ptr = reinterpret_cast<const uint64_t*>(start);
    while (reinterpret_cast<const char*>(ptr + 1) <= start + len) {
        hasher.feed(*ptr);
        ++ptr;
    }

    auto cptr = reinterpret_cast<const char*>(ptr);
    if (cptr < start + len) {
        uint64_t v = 0;
        std::copy(cptr, start + len, reinterpret_cast<char*>(&v));
        hasher.feed(v);
    }

    printf("%llx\n", static_cast<unsigned long long>(hasher.get()));
    return 0;
}

std::unordered_map<std::string, CommandHandler> commands = {
        {"encrypt_predefined_rc4", command_encrypt_predefined_rc4},
        {"encrypt_rc4", command_encrypt_rc4},
        {"encrypt_predefined_sfrc4", command_encrypt_predefined_sfrc4},
        {"encrypt_sfrc4", command_encrypt_sfrc4},
        {"hash", command_hash},
};

int main(int argc, char** argv) {
    if (argc == 1) {
        printf("%s", usage);
        return 1;
    }

    auto it = commands.find(argv[1]);
    if (it == commands.end()) {
        printf("Invalid command arguments.\n");
        printf("%s", usage);
        return 1;
    }
    return it->second(argc, argv);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
