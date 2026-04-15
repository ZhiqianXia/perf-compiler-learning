// eml_sheffer.cpp — The EML Sheffer Operator: All Elementary Functions from a Single Operator
// Based on: "All elementary functions from a single operator" (arXiv:2603.21852)
//
// The operator eml(x,y) = exp(x) - ln(y), together with the constant 1,
// generates all standard elementary functions: arithmetic, transcendental,
// and fundamental constants (e, π, i).
//
// Grammar: S → 1 | eml(S, S)
//
// Compile: g++ -std=c++17 -O2 -o eml_sheffer eml_sheffer.cpp -lm
// Run:     ./eml_sheffer

#include <complex>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <functional>
#include <vector>
#include <cstdlib>

using Complex = std::complex<double>;

// ============================================================
// The Single Binary Operator
// eml(x, y) = exp(x) - ln(y)
// Operates over ℂ using the principal branch of complex log.
// Relies on IEEE 754 extended reals: ln(0) = -∞, exp(-∞) = 0.
// ============================================================
static inline Complex eml(Complex x, Complex y) {
    return std::exp(x) - std::log(y);
}

// The only distinguished constant
static const Complex ONE(1.0, 0.0);

// ============================================================
// Bootstrapping chain: deriving all elementary functions
// using only eml() and the constant 1.
//
// Each level uses only primitives from previous levels.
// The chain mirrors the paper's Figure 1 discovery order.
// ============================================================

// --- Level 0: Constants (depth 1) ---

// e = eml(1, 1) = exp(1) - ln(1) = e - 0 = e
static Complex eml_e() {
    return eml(ONE, ONE);
}

// --- Level 1: exp and ln (depth 1-3) ---

// exp(x) = eml(x, 1) = exp(x) - ln(1) = exp(x) - 0 = exp(x)
// RPN: x 1 E  (K = 3)
static Complex eml_exp(Complex x) {
    return eml(x, ONE);
}

// ln(x) = eml(1, eml(eml(1, x), 1))
// Proof: eml(1, x) = e - ln(x)
//        eml(e - ln(x), 1) = exp(e - ln(x)) = e^e / x
//        eml(1, e^e / x) = e - ln(e^e / x) = e - (e - ln(x)) = ln(x)
// RPN: 1 1 x E 1 E E  (K = 7)
static Complex eml_ln(Complex x) {
    return eml(ONE, eml(eml(ONE, x), ONE));
}

// --- Level 2: Subtraction (depth depends on operands) ---

// x - y = eml(ln(x), exp(y)) = exp(ln(x)) - ln(exp(y)) = x - y
// RPN: 1 1 x E 1 E E y 1 E E  (K = 11)
static Complex eml_sub(Complex x, Complex y) {
    return eml(eml_ln(x), eml_exp(y));
}

// --- Level 3: Zero (K = 7) ---

// 0 = ln(1) computed via the eml_ln formula
// 0 = eml(1, eml(eml(1, 1), 1))
//   = eml(1, eml(e, 1))
//   = eml(1, exp(e))
//   = e - ln(exp(e))
//   = e - e = 0
static Complex eml_zero() {
    return eml_ln(ONE);
}

// --- Level 4: Negation ---

// -x = 0 - x = eml(ln(0), exp(x))
// ln(0) = -∞ (IEEE 754), exp(-∞) = 0
// So: eml(-∞, exp(x)) = 0 - ln(exp(x)) = 0 - x = -x
static Complex eml_neg(Complex x) {
    return eml_sub(eml_zero(), x);
}

// --- Level 5: Addition ---

// x + y = x - (-y)
static Complex eml_add(Complex x, Complex y) {
    return eml_sub(x, eml_neg(y));
}

// --- Level 6: Multiplication and Division ---

// x * y = exp(ln(x) + ln(y))
static Complex eml_mul(Complex x, Complex y) {
    return eml_exp(eml_add(eml_ln(x), eml_ln(y)));
}

// 1/x = exp(-ln(x))
static Complex eml_inv(Complex x) {
    return eml_exp(eml_neg(eml_ln(x)));
}

// x / y = x * (1/y)
static Complex eml_div(Complex x, Complex y) {
    return eml_mul(x, eml_inv(y));
}

// --- Level 7: Integer constants ---

// -1 = 0 - 1
static Complex eml_neg_one() {
    return eml_neg(ONE);
}

// 2 = 1 + 1
static Complex eml_two() {
    return eml_add(ONE, ONE);
}

// 1/2
static Complex eml_half() {
    return eml_inv(eml_two());
}

// --- Level 8: Transcendental constants ---

// iπ = ln(-1)   (principal branch: ln(-1) = iπ)
static Complex eml_i_pi() {
    return eml_ln(eml_neg_one());
}

// π² = -(iπ)²  since (iπ)² = i²π² = -π², so -(iπ)² = π²
static Complex eml_pi_squared() {
    return eml_neg(eml_mul(eml_i_pi(), eml_i_pi()));
}

// π = √(π²) = exp(ln(π²) / 2) = exp(ln(π²) * (1/2))
static Complex eml_pi() {
    return eml_exp(eml_mul(eml_ln(eml_pi_squared()), eml_half()));
}

// i = (iπ) / π
static Complex eml_i() {
    return eml_div(eml_i_pi(), eml_pi());
}

// --- Level 9: Power and root ---

// x^y = exp(y * ln(x))
static Complex eml_pow(Complex x, Complex y) {
    return eml_exp(eml_mul(y, eml_ln(x)));
}

// √x = x^(1/2) = exp(ln(x) * (1/2))
static Complex eml_sqrt(Complex x) {
    return eml_pow(x, eml_half());
}

// x² = x * x
static Complex eml_sqr(Complex x) {
    return eml_mul(x, x);
}

// --- Level 10: Trigonometric functions (via Euler's formula) ---

// sin(x) = (exp(ix) - exp(-ix)) / (2i)
static Complex eml_sin(Complex x) {
    Complex i = eml_i();
    Complex ix = eml_mul(i, x);
    Complex neg_ix = eml_neg(ix);
    Complex num = eml_sub(eml_exp(ix), eml_exp(neg_ix));
    Complex denom = eml_mul(eml_two(), i);
    return eml_div(num, denom);
}

// cos(x) = (exp(ix) + exp(-ix)) / 2
static Complex eml_cos(Complex x) {
    Complex i = eml_i();
    Complex ix = eml_mul(i, x);
    Complex neg_ix = eml_neg(ix);
    Complex num = eml_add(eml_exp(ix), eml_exp(neg_ix));
    return eml_div(num, eml_two());
}

// tan(x) = sin(x) / cos(x)
static Complex eml_tan(Complex x) {
    return eml_div(eml_sin(x), eml_cos(x));
}

// --- Level 11: Hyperbolic functions ---

// sinh(x) = (exp(x) - exp(-x)) / 2
static Complex eml_sinh(Complex x) {
    Complex num = eml_sub(eml_exp(x), eml_exp(eml_neg(x)));
    return eml_div(num, eml_two());
}

// cosh(x) = (exp(x) + exp(-x)) / 2
static Complex eml_cosh(Complex x) {
    Complex num = eml_add(eml_exp(x), eml_exp(eml_neg(x)));
    return eml_div(num, eml_two());
}

// tanh(x) = sinh(x) / cosh(x)
static Complex eml_tanh(Complex x) {
    return eml_div(eml_sinh(x), eml_cosh(x));
}

// --- Level 12: Inverse trigonometric functions ---

// arcsin(x) = -i * ln(ix + √(1 - x²))
static Complex eml_arcsin(Complex x) {
    Complex i = eml_i();
    Complex neg_i = eml_neg(i);
    Complex ix = eml_mul(i, x);
    Complex one_minus_x2 = eml_sub(ONE, eml_sqr(x));
    Complex root = eml_sqrt(one_minus_x2);
    return eml_mul(neg_i, eml_ln(eml_add(ix, root)));
}

// arccos(x) = -i * ln(x + √(x² - 1))
static Complex eml_arccos(Complex x) {
    Complex i = eml_i();
    Complex neg_i = eml_neg(i);
    Complex x2_minus_1 = eml_sub(eml_sqr(x), ONE);
    Complex root = eml_sqrt(x2_minus_1);
    return eml_mul(neg_i, eml_ln(eml_add(x, root)));
}

// arctan(x) = (i/2) * ln((1 - ix) / (1 + ix))
static Complex eml_arctan(Complex x) {
    Complex i = eml_i();
    Complex i_half = eml_mul(i, eml_half());
    Complex ix = eml_mul(i, x);
    Complex num = eml_sub(ONE, ix);
    Complex denom = eml_add(ONE, ix);
    return eml_mul(i_half, eml_ln(eml_div(num, denom)));
}

// --- Level 13: Inverse hyperbolic functions ---

// arsinh(x) = ln(x + √(x² + 1))
static Complex eml_arsinh(Complex x) {
    Complex x2_plus_1 = eml_add(eml_sqr(x), ONE);
    return eml_ln(eml_add(x, eml_sqrt(x2_plus_1)));
}

// arcosh(x) = ln(x + √(x² - 1))
static Complex eml_arcosh(Complex x) {
    Complex x2_minus_1 = eml_sub(eml_sqr(x), ONE);
    return eml_ln(eml_add(x, eml_sqrt(x2_minus_1)));
}

// artanh(x) = (1/2) * ln((1 + x) / (1 - x))
static Complex eml_artanh(Complex x) {
    Complex num = eml_add(ONE, x);
    Complex denom = eml_sub(ONE, x);
    return eml_mul(eml_half(), eml_ln(eml_div(num, denom)));
}

// --- Level 14: Misc functions from Table 1 ---

// σ(x) = 1 / (1 + exp(-x))   (logistic sigmoid)
static Complex eml_sigmoid(Complex x) {
    return eml_inv(eml_add(ONE, eml_exp(eml_neg(x))));
}

// log_x(y) = ln(y) / ln(x)   (arbitrary-base logarithm)
static Complex eml_log_base(Complex x, Complex y) {
    return eml_div(eml_ln(y), eml_ln(x));
}

// avg(x, y) = (x + y) / 2
static Complex eml_avg(Complex x, Complex y) {
    return eml_mul(eml_add(x, y), eml_half());
}

// hypot(x, y) = √(x² + y²)
static Complex eml_hypot(Complex x, Complex y) {
    return eml_sqrt(eml_add(eml_sqr(x), eml_sqr(y)));
}

// ============================================================
// Pure EML forms (expanded to only eml() and 1)
// These show the raw tree structure: S → 1 | eml(S,S)
// ============================================================

// exp(x) in pure form: eml(x, 1)                              K=3
static Complex pure_eml_exp(Complex x) {
    return eml(x, ONE);
}

// e in pure form: eml(1, 1)                                    K=3
static Complex pure_eml_e() {
    return eml(ONE, ONE);
}

// ln(x) in pure form: eml(1, eml(eml(1, x), 1))              K=7
static Complex pure_eml_ln(Complex x) {
    return eml(ONE, eml(eml(ONE, x), ONE));
}

// 0 in pure form: eml(1, eml(eml(1, 1), 1))                  K=7
static Complex pure_eml_zero() {
    return eml(ONE, eml(eml(ONE, ONE), ONE));
}

// x - y in pure form (K=11):
// eml(eml(1, eml(eml(1, x), 1)), eml(y, 1))
static Complex pure_eml_sub(Complex x, Complex y) {
    return eml(eml(ONE, eml(eml(ONE, x), ONE)), eml(y, ONE));
}

// x + 1 in pure form (K=19):
// Since x + 1 = x - (-1) = x - (0 - 1),
// and 0 = eml(1, eml(eml(1,1),1)),
// we use the sub formula with y = -1
// For brevity, use bootstrapped form:
static Complex pure_eml_x_plus_1(Complex x) {
    Complex zero = eml(ONE, eml(eml(ONE, ONE), ONE));
    Complex neg_one = eml(eml(ONE, eml(eml(ONE, zero), ONE)), eml(ONE, ONE));
    Complex neg_neg_one = eml(eml(ONE, eml(eml(ONE, neg_one), ONE)),
                               eml(neg_one, ONE));
    return eml(eml(ONE, eml(eml(ONE, x), ONE)),
               eml(neg_neg_one, ONE));
}

// ============================================================
// Verification
// ============================================================

struct TestResult {
    std::string name;
    Complex eml_val;
    Complex ref_val;
    double error;
    bool passed;
};

static double rel_error(Complex a, Complex b) {
    double denom = std::abs(b);
    if (denom < 1e-15) denom = 1.0;
    return std::abs(a - b) / denom;
}

static void print_complex(const std::string& label, Complex z) {
    if (std::abs(z.imag()) < 1e-12) {
        std::cout << std::setw(12) << z.real();
    } else if (std::abs(z.real()) < 1e-12) {
        std::cout << std::setw(12) << z.imag() << "i";
    } else {
        std::cout << z.real() << std::showpos << z.imag() << "i" << std::noshowpos;
    }
}

int main() {
    constexpr double TOL = 1e-10;
    int pass_count = 0, fail_count = 0;

    std::cout << "============================================================\n";
    std::cout << " EML Sheffer Operator: eml(x,y) = exp(x) - ln(y)\n";
    std::cout << " All elementary functions from a single binary operator + 1\n";
    std::cout << " arXiv:2603.21852\n";
    std::cout << "============================================================\n\n";

    // Test values
    Complex x(1.7, 0.0);
    Complex y(2.3, 0.0);

    auto check = [&](const std::string& name, Complex eml_val, Complex ref_val) {
        double err = rel_error(eml_val, ref_val);
        bool ok = err < TOL;
        if (ok) pass_count++; else fail_count++;
        std::cout << (ok ? "[PASS]" : "[FAIL]") << "  " << std::left << std::setw(24) << name;
        std::cout << "  EML=";
        print_complex("", eml_val);
        std::cout << "  REF=";
        print_complex("", ref_val);
        std::cout << "  err=" << std::scientific << std::setprecision(2) << err;
        std::cout << std::fixed << "\n";
    };

    // --- Constants ---
    std::cout << "--- Constants ---\n";
    check("e",           eml_e(),         Complex(M_E, 0));
    check("0",           eml_zero(),      Complex(0, 0));
    check("-1",          eml_neg_one(),    Complex(-1, 0));
    check("2",           eml_two(),        Complex(2, 0));
    check("1/2",         eml_half(),       Complex(0.5, 0));
    check("i*pi",        eml_i_pi(),       Complex(0, M_PI));
    check("pi",          eml_pi(),         Complex(M_PI, 0));
    check("i",           eml_i(),          Complex(0, 1));
    std::cout << "\n";

    // --- Unary functions (on x = 1.7) ---
    std::cout << "--- Unary functions (x = " << x.real() << ") ---\n";
    check("exp(x)",      eml_exp(x),      std::exp(x));
    check("ln(x)",       eml_ln(x),       std::log(x));
    check("-x",          eml_neg(x),      -x);
    check("1/x",         eml_inv(x),      1.0/x);
    check("x^2",         eml_sqr(x),      x*x);
    check("sqrt(x)",     eml_sqrt(x),     std::sqrt(x));
    check("sigmoid(x)",  eml_sigmoid(x),  1.0/(1.0+std::exp(-x)));
    std::cout << "\n";

    // --- Arithmetic (x = 1.7, y = 2.3) ---
    std::cout << "--- Arithmetic (x = " << x.real() << ", y = " << y.real() << ") ---\n";
    check("x - y",       eml_sub(x, y),   x - y);
    check("x + y",       eml_add(x, y),   x + y);
    check("x * y",       eml_mul(x, y),   x * y);
    check("x / y",       eml_div(x, y),   x / y);
    check("x^y",         eml_pow(x, y),   std::pow(x, y));
    check("log_x(y)",    eml_log_base(x, y), std::log(y)/std::log(x));
    check("avg(x,y)",    eml_avg(x, y),   (x+y)/2.0);
    check("hypot(x,y)",  eml_hypot(x, y), std::sqrt(x*x + y*y));
    std::cout << "\n";

    // --- Trigonometric (x = 0.7 to stay in domain) ---
    Complex t(0.7, 0.0);
    std::cout << "--- Trigonometric (t = " << t.real() << ") ---\n";
    check("sin(t)",      eml_sin(t),      std::sin(t));
    check("cos(t)",      eml_cos(t),      std::cos(t));
    check("tan(t)",      eml_tan(t),      std::tan(t));
    check("arcsin(t)",   eml_arcsin(t),   std::asin(t));
    check("arccos(t)",   eml_arccos(t),   std::acos(t));
    check("arctan(t)",   eml_arctan(t),   std::atan(t));
    std::cout << "\n";

    // --- Hyperbolic (x = 1.7) ---
    std::cout << "--- Hyperbolic (x = " << x.real() << ") ---\n";
    check("sinh(x)",     eml_sinh(x),     std::sinh(x));
    check("cosh(x)",     eml_cosh(x),     std::cosh(x));
    check("tanh(x)",     eml_tanh(x),     std::tanh(x));
    check("arsinh(x)",   eml_arsinh(x),   std::asinh(x));

    Complex c(1.7, 0.0);
    check("arcosh(c)",   eml_arcosh(c),   std::acosh(c));

    Complex h(0.7, 0.0);
    check("artanh(h)",   eml_artanh(h),   std::atanh(h));
    std::cout << "\n";

    // --- Pure EML forms (no bootstrapping, only eml and 1) ---
    std::cout << "--- Pure EML forms (only eml() and 1) ---\n";
    check("pure e",      pure_eml_e(),      Complex(M_E, 0));
    check("pure exp(x)", pure_eml_exp(x),   std::exp(x));
    check("pure ln(x)",  pure_eml_ln(x),    std::log(x));
    check("pure 0",      pure_eml_zero(),   Complex(0, 0));
    check("pure x-y",    pure_eml_sub(x, y), x - y);
    std::cout << "\n";

    // --- Euler's identity: e^(iπ) + 1 = 0 ---
    std::cout << "--- Euler's identity ---\n";
    Complex euler = eml_add(eml_exp(eml_mul(eml_i(), eml_pi())), ONE);
    check("e^(i*pi)+1=0", euler, Complex(0, 0));
    std::cout << "\n";

    // --- Summary ---
    std::cout << "============================================================\n";
    std::cout << " Results: " << pass_count << " passed, " << fail_count << " failed"
              << " (tolerance: " << std::scientific << TOL << ")\n";
    std::cout << "============================================================\n";

    return fail_count > 0 ? 1 : 0;
}
