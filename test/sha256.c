/* sha256.c - SHA reference implementation using C            */
/*   Written and placed in public domain by Jeffrey Walton    */

/* xlc -DTEST_MAIN sha256.c -o sha256.exe           */
/* gcc -DTEST_MAIN -std=c99 sha256.c -o sha256.exe  */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const uint32_t K256[]
    = {0x428A'2F98, 0x7137'4491, 0xB5C0'FBCF, 0xE9B5'DBA5, 0x3956'C25B, 0x59F1'11F1, 0x923F'82A4, 0xAB1C'5ED5,
       0xD807'AA98, 0x1283'5B01, 0x2431'85BE, 0x550C'7DC3, 0x72BE'5D74, 0x80DE'B1FE, 0x9BDC'06A7, 0xC19B'F174,
       0xE49B'69C1, 0xEFBE'4786, 0x0FC1'9DC6, 0x240C'A1CC, 0x2DE9'2C6F, 0x4A74'84AA, 0x5CB0'A9DC, 0x76F9'88DA,
       0x983E'5152, 0xA831'C66D, 0xB003'27C8, 0xBF59'7FC7, 0xC6E0'0BF3, 0xD5A7'9147, 0x06CA'6351, 0x1429'2967,
       0x27B7'0A85, 0x2E1B'2138, 0x4D2C'6DFC, 0x5338'0D13, 0x650A'7354, 0x766A'0ABB, 0x81C2'C92E, 0x9272'2C85,
       0xA2BF'E8A1, 0xA81A'664B, 0xC24B'8B70, 0xC76C'51A3, 0xD192'E819, 0xD699'0624, 0xF40E'3585, 0x106A'A070,
       0x19A4'C116, 0x1E37'6C08, 0x2748'774C, 0x34B0'BCB5, 0x391C'0CB3, 0x4ED8'AA4A, 0x5B9C'CA4F, 0x682E'6FF3,
       0x748F'82EE, 0x78A5'636F, 0x84C8'7814, 0x8CC7'0208, 0x90BE'FFFA, 0xA450'6CEB, 0xBEF9'A3F7, 0xC671'78F2};

#define ROTATE(x, y) (((x) >> (y)) | ((x) << (32 - (y))))
#define Sigma0(x) (ROTATE((x), 2) ^ ROTATE((x), 13) ^ ROTATE((x), 22))
#define Sigma1(x) (ROTATE((x), 6) ^ ROTATE((x), 11) ^ ROTATE((x), 25))
#define sigma0(x) (ROTATE((x), 7) ^ ROTATE((x), 18) ^ ((x) >> 3))
#define sigma1(x) (ROTATE((x), 17) ^ ROTATE((x), 19) ^ ((x) >> 10))

#define Ch(x, y, z) (((x) & (y)) ^ ((~(x)) & (z)))
#define Maj(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

/* Avoid undefined behavior                    */
/* https://stackoverflow.com/q/29538935/608639 */
uint32_t B2U32(uint8_t val, uint8_t sh)
{
    return ((uint32_t) val) << sh;
}

/* Process multiple blocks. The caller is responsible for setting the initial */
/*  state, and the caller is responsible for padding the final block.        */
void sha256_process(uint32_t state[8], const uint8_t data[], uint32_t length)
{
    uint32_t a, b, c, d, e, f, g, h, s0, s1, T1, T2;
    uint32_t X[16], i;

    size_t blocks = length / 64;
    while(blocks--)
    {
        a = state[0];
        b = state[1];
        c = state[2];
        d = state[3];
        e = state[4];
        f = state[5];
        g = state[6];
        h = state[7];

        for(i = 0; i < 16; i++)
        {
            X[i] = B2U32(data[0], 24) | B2U32(data[1], 16) | B2U32(data[2], 8) | B2U32(data[3], 0);
            data += 4;

            T1 = h;
            T1 += Sigma1(e);
            T1 += Ch(e, f, g);
            T1 += K256[i];
            T1 += X[i];

            T2 = Sigma0(a);
            T2 += Maj(a, b, c);

            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
        }

        for(; i < 64; i++)
        {
            s0 = X[(i + 1) & 0x0f];
            s0 = sigma0(s0);
            s1 = X[(i + 14) & 0x0f];
            s1 = sigma1(s1);

            T1 = X[i & 0xf] += s0 + s1 + X[(i + 9) & 0xf];
            T1 += h + Sigma1(e) + Ch(e, f, g) + K256[i];
            T2 = Sigma0(a) + Maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }
}
