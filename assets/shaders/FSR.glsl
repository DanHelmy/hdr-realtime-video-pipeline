// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// FidelityFX FSR v1.0.2 EASU + RCAS by AMD, adapted for this app's RGB48 mpv feed.
// The original mpv port operated on LUMA only. This version hooks MAIN so it
// runs after conversion to RGB and before mpv's scaler, which is the correct
// resizable hook point for raw RGB video.

//!HOOK MAIN
//!BIND HOOKED
//!SAVE EASUTEX
//!DESC FidelityFX Super Resolution v1.0.2 RGB (EASU)
//!WHEN OUTPUT.w OUTPUT.h * MAIN.w MAIN.h * / 1.0 >
//!WIDTH OUTPUT.w OUTPUT.w MAIN.w 2 * < * MAIN.w 2 * OUTPUT.w MAIN.w 2 * > * + OUTPUT.w OUTPUT.w MAIN.w 2 * = * +
//!HEIGHT OUTPUT.h OUTPUT.h MAIN.h 2 * < * MAIN.h 2 * OUTPUT.h MAIN.h 2 * > * + OUTPUT.h OUTPUT.h MAIN.h 2 * = * +
//!COMPONENTS 3

// User variables - EASU
#define FSR_PQ 0
#define FSR_EASU_DERING 1
#define FSR_EASU_SIMPLE_ANALYSIS 0
#define FSR_EASU_QUIT_EARLY 0

#ifndef FSR_EASU_DIR_THRESHOLD
    #if (FSR_EASU_QUIT_EARLY == 1)
        #define FSR_EASU_DIR_THRESHOLD 64.0
    #elif (FSR_EASU_QUIT_EARLY == 0)
        #define FSR_EASU_DIR_THRESHOLD 32768.0
    #endif
#endif

float APrxLoRcpF1(float a) {
    return uintBitsToFloat(uint(0x7ef07ebb) - floatBitsToUint(a));
}

float APrxLoRsqF1(float a) {
    return uintBitsToFloat(uint(0x5f347d74) - (floatBitsToUint(a) >> uint(1)));
}

float AMin3F1(float x, float y, float z) {
    return min(x, min(y, z));
}

float AMax3F1(float x, float y, float z) {
    return max(x, max(y, z));
}

vec3 AMin3F3(vec3 x, vec3 y, vec3 z) {
    return min(x, min(y, z));
}

vec3 AMax3F3(vec3 x, vec3 y, vec3 z) {
    return max(x, max(y, z));
}

float FsrLuma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

#if (FSR_PQ == 1)

vec3 ToGamma2(vec3 a) {
    return pow(max(a, vec3(0.0)), vec3(4.0));
}

#endif

void FsrEasuTap(
    inout vec3 aC,
    inout float aW,
    vec2 off,
    vec2 dir,
    vec2 len,
    float lob,
    float clp,
    vec3 c
) {
    vec2 v;
    v.x = (off.x * dir.x) + (off.y * dir.y);
    v.y = (off.x * -dir.y) + (off.y * dir.x);
    v *= len;
    float d2 = v.x * v.x + v.y * v.y;
    d2 = min(d2, clp);
    float wB = float(2.0 / 5.0) * d2 + -1.0;
    float wA = lob * d2 + -1.0;
    wB *= wB;
    wA *= wA;
    wB = float(25.0 / 16.0) * wB + float(-(25.0 / 16.0 - 1.0));
    float w = wB * wA;
    aC += c * w;
    aW += w;
}

void FsrEasuSet(
    inout vec2 dir,
    inout float len,
    vec2 pp,
#if (FSR_EASU_SIMPLE_ANALYSIS == 1)
    float b, float c,
    float i, float j, float f, float e,
    float k, float l, float h, float g,
    float o, float n
#elif (FSR_EASU_SIMPLE_ANALYSIS == 0)
    bool biS, bool biT, bool biU, bool biV,
    float lA, float lB, float lC, float lD, float lE
#endif
) {
#if (FSR_EASU_SIMPLE_ANALYSIS == 1)
    vec4 w = vec4(0.0);
    w.x = (1.0 - pp.x) * (1.0 - pp.y);
    w.y =        pp.x  * (1.0 - pp.y);
    w.z = (1.0 - pp.x) *        pp.y;
    w.w =        pp.x  *        pp.y;

    float lA = dot(w, vec4(b, c, f, g));
    float lB = dot(w, vec4(e, f, i, j));
    float lC = dot(w, vec4(f, g, j, k));
    float lD = dot(w, vec4(g, h, k, l));
    float lE = dot(w, vec4(j, k, n, o));
#elif (FSR_EASU_SIMPLE_ANALYSIS == 0)
    float w = 0.0;
    if (biS)
        w = (1.0 - pp.x) * (1.0 - pp.y);
    if (biT)
        w =        pp.x  * (1.0 - pp.y);
    if (biU)
        w = (1.0 - pp.x) *        pp.y;
    if (biV)
        w =        pp.x  *        pp.y;
#endif

    float dc = lD - lC;
    float cb = lC - lB;
    float lenX = max(abs(dc), abs(cb));
    lenX = APrxLoRcpF1(lenX);
    float dirX = lD - lB;
    lenX = clamp(abs(dirX) * lenX, 0.0, 1.0);
    lenX *= lenX;

    float ec = lE - lC;
    float ca = lC - lA;
    float lenY = max(abs(ec), abs(ca));
    lenY = APrxLoRcpF1(lenY);
    float dirY = lE - lA;
    lenY = clamp(abs(dirY) * lenY, 0.0, 1.0);
    lenY *= lenY;
#if (FSR_EASU_SIMPLE_ANALYSIS == 1)
    len = lenX + lenY;
    dir = vec2(dirX, dirY);
#elif (FSR_EASU_SIMPLE_ANALYSIS == 0)
    dir += vec2(dirX, dirY) * w;
    len += dot(vec2(w), vec2(lenX, lenY));
#endif
}

vec4 hook() {
    vec2 pp = HOOKED_pos * HOOKED_size - vec2(0.5);
    vec2 fp = floor(pp);
    pp -= fp;

    vec3 bC = HOOKED_tex(vec2((fp + vec2( 0.5, -0.5)) * HOOKED_pt)).rgb;
    vec3 cC = HOOKED_tex(vec2((fp + vec2( 1.5, -0.5)) * HOOKED_pt)).rgb;
    vec3 eC = HOOKED_tex(vec2((fp + vec2(-0.5,  0.5)) * HOOKED_pt)).rgb;
    vec3 fC = HOOKED_tex(vec2((fp + vec2( 0.5,  0.5)) * HOOKED_pt)).rgb;
    vec3 gC = HOOKED_tex(vec2((fp + vec2( 1.5,  0.5)) * HOOKED_pt)).rgb;
    vec3 hC = HOOKED_tex(vec2((fp + vec2( 2.5,  0.5)) * HOOKED_pt)).rgb;
    vec3 iC = HOOKED_tex(vec2((fp + vec2(-0.5,  1.5)) * HOOKED_pt)).rgb;
    vec3 jC = HOOKED_tex(vec2((fp + vec2( 0.5,  1.5)) * HOOKED_pt)).rgb;
    vec3 kC = HOOKED_tex(vec2((fp + vec2( 1.5,  1.5)) * HOOKED_pt)).rgb;
    vec3 lC = HOOKED_tex(vec2((fp + vec2( 2.5,  1.5)) * HOOKED_pt)).rgb;
    vec3 nC = HOOKED_tex(vec2((fp + vec2( 0.5,  2.5)) * HOOKED_pt)).rgb;
    vec3 oC = HOOKED_tex(vec2((fp + vec2( 1.5,  2.5)) * HOOKED_pt)).rgb;

#if (FSR_PQ == 1)
    bC = ToGamma2(bC);
    cC = ToGamma2(cC);
    eC = ToGamma2(eC);
    fC = ToGamma2(fC);
    gC = ToGamma2(gC);
    hC = ToGamma2(hC);
    iC = ToGamma2(iC);
    jC = ToGamma2(jC);
    kC = ToGamma2(kC);
    lC = ToGamma2(lC);
    nC = ToGamma2(nC);
    oC = ToGamma2(oC);
#endif

    float bL = FsrLuma(bC);
    float cL = FsrLuma(cC);
    float eL = FsrLuma(eC);
    float fL = FsrLuma(fC);
    float gL = FsrLuma(gC);
    float hL = FsrLuma(hC);
    float iL = FsrLuma(iC);
    float jL = FsrLuma(jC);
    float kL = FsrLuma(kC);
    float lL = FsrLuma(lC);
    float nL = FsrLuma(nC);
    float oL = FsrLuma(oC);

    vec2 dir = vec2(0.0);
    float len = 0.0;
#if (FSR_EASU_SIMPLE_ANALYSIS == 1)
    FsrEasuSet(dir, len, pp, bL, cL, iL, jL, fL, eL, kL, lL, hL, gL, oL, nL);
#elif (FSR_EASU_SIMPLE_ANALYSIS == 0)
    FsrEasuSet(dir, len, pp, true, false, false, false, bL, eL, fL, gL, jL);
    FsrEasuSet(dir, len, pp, false, true, false, false, cL, fL, gL, hL, kL);
    FsrEasuSet(dir, len, pp, false, false, true, false, fL, iL, jL, kL, nL);
    FsrEasuSet(dir, len, pp, false, false, false, true, gL, jL, kL, lL, oL);
#endif

    vec2 dir2 = dir * dir;
    float dirR = dir2.x + dir2.y;
    bool zro = dirR < float(1.0 / FSR_EASU_DIR_THRESHOLD);
    dirR = APrxLoRsqF1(dirR);
#if (FSR_EASU_QUIT_EARLY == 1)
    if (zro) {
        vec4 w = vec4(0.0);
        w.x = (1.0 - pp.x) * (1.0 - pp.y);
        w.y =        pp.x  * (1.0 - pp.y);
        w.z = (1.0 - pp.x) *        pp.y;
        w.w =        pp.x  *        pp.y;
        vec3 bilinear = fC * w.x + gC * w.y + jC * w.z + kC * w.w;
        return vec4(clamp(bilinear, vec3(0.0), vec3(1.0)), 1.0);
    }
#elif (FSR_EASU_QUIT_EARLY == 0)
    dirR = zro ? 1.0 : dirR;
    dir.x = zro ? 1.0 : dir.x;
#endif
    dir *= vec2(dirR);

    len = len * 0.5;
    len *= len;
    float stretch = (dir.x * dir.x + dir.y * dir.y) * APrxLoRcpF1(max(abs(dir.x), abs(dir.y)));
    vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 + -0.5 * len);
    float lob = 0.5 + float((1.0 / 4.0 - 0.04) - 0.5) * len;
    float clp = APrxLoRcpF1(lob);

    vec3 aC = vec3(0.0);
    float aW = 0.0;
    FsrEasuTap(aC, aW, vec2( 0.0,-1.0) - pp, dir, len2, lob, clp, bC);
    FsrEasuTap(aC, aW, vec2( 1.0,-1.0) - pp, dir, len2, lob, clp, cC);
    FsrEasuTap(aC, aW, vec2(-1.0, 1.0) - pp, dir, len2, lob, clp, iC);
    FsrEasuTap(aC, aW, vec2( 0.0, 1.0) - pp, dir, len2, lob, clp, jC);
    FsrEasuTap(aC, aW, vec2( 0.0, 0.0) - pp, dir, len2, lob, clp, fC);
    FsrEasuTap(aC, aW, vec2(-1.0, 0.0) - pp, dir, len2, lob, clp, eC);
    FsrEasuTap(aC, aW, vec2( 1.0, 1.0) - pp, dir, len2, lob, clp, kC);
    FsrEasuTap(aC, aW, vec2( 2.0, 1.0) - pp, dir, len2, lob, clp, lC);
    FsrEasuTap(aC, aW, vec2( 2.0, 0.0) - pp, dir, len2, lob, clp, hC);
    FsrEasuTap(aC, aW, vec2( 1.0, 0.0) - pp, dir, len2, lob, clp, gC);
    FsrEasuTap(aC, aW, vec2( 1.0, 2.0) - pp, dir, len2, lob, clp, oC);
    FsrEasuTap(aC, aW, vec2( 0.0, 2.0) - pp, dir, len2, lob, clp, nC);

    vec3 pix = aC / aW;
#if (FSR_EASU_DERING == 1)
    vec3 min1 = min(AMin3F3(fC, gC, jC), kC);
    vec3 max1 = max(AMax3F3(fC, gC, jC), kC);
    pix = clamp(pix, min1, max1);
#endif
    return vec4(clamp(pix, vec3(0.0), vec3(1.0)), 1.0);
}

//!HOOK MAIN
//!BIND EASUTEX
//!DESC FidelityFX Super Resolution v1.0.2 RGB (RCAS)
//!WIDTH EASUTEX.w
//!HEIGHT EASUTEX.h
//!COMPONENTS 3

#define FSR_RCAS_SHARPNESS 0.2
#define FSR_RCAS_DENOISE 1
#define FSR_RCAS_LIMIT (0.25 - (1.0 / 16.0))

float FsrRcasRcp(float a) {
    float b = uintBitsToFloat(uint(0x7ef19fff) - floatBitsToUint(a));
    return b * (-b * a + 2.0);
}

float FsrRcasLuma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

#if (FSR_PQ == 1)

vec3 FsrRcasFromGamma2(vec3 a) {
    return sqrt(sqrt(max(a, vec3(0.0))));
}

#endif

vec4 hook() {
    vec3 b = EASUTEX_texOff(vec2( 0.0, -1.0)).rgb;
    vec3 d = EASUTEX_texOff(vec2(-1.0,  0.0)).rgb;
    vec3 e = EASUTEX_tex(EASUTEX_pos).rgb;
    vec3 f = EASUTEX_texOff(vec2( 1.0,  0.0)).rgb;
    vec3 h = EASUTEX_texOff(vec2( 0.0,  1.0)).rgb;

    float bL = FsrRcasLuma(b);
    float dL = FsrRcasLuma(d);
    float eL = FsrRcasLuma(e);
    float fL = FsrRcasLuma(f);
    float hL = FsrRcasLuma(h);

    float mn1L = min(AMin3F1(bL, dL, fL), hL);
    float mx1L = max(AMax3F1(bL, dL, fL), hL);
    float hitMinL = min(mn1L, eL) / max(4.0 * mx1L, 1e-6);
    float hitMaxL = (1.0 - max(mx1L, eL)) / min(4.0 * mn1L - 4.0, -1e-6);
    float lobeL = max(-hitMinL, hitMaxL);
    float lobe = max(float(-FSR_RCAS_LIMIT), min(lobeL, 0.0));
    lobe *= exp2(-clamp(float(FSR_RCAS_SHARPNESS), 0.0, 2.0));

#if (FSR_RCAS_DENOISE == 1)
    float nz = 0.25 * (bL + dL + fL + hL) - eL;
    float rangeL = max(
        AMax3F1(AMax3F1(bL, dL, eL), fL, hL) - AMin3F1(AMin3F1(bL, dL, eL), fL, hL),
        1e-6
    );
    nz = clamp(abs(nz) * FsrRcasRcp(rangeL), 0.0, 1.0);
    lobe *= -0.5 * nz + 1.0;
#endif

    float rcpL = FsrRcasRcp(4.0 * lobe + 1.0);
    vec3 pix = (lobe * b + lobe * d + lobe * h + lobe * f + e) * rcpL;

#if (FSR_PQ == 1)
    pix = FsrRcasFromGamma2(pix);
#endif

    return vec4(clamp(pix, vec3(0.0), vec3(1.0)), 1.0);
}
