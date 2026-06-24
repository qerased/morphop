#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

using uchar = unsigned char;
namespace fs = std::filesystem;

enum class Cell { Zero, One, DontCare };

struct Kernel3x3 {
    std::array<Cell, 9> cells{};
};

struct Metrics {
    int pixels = 0;
    int components8 = 0;
    int changed = 0;
    int added = 0;
    int removed = 0;
    int endpoints = 0;
};

static constexpr int kCenter = 4;
static const std::array<std::pair<int, int>, 9> kOffsets = {{
    {-1, -1}, {0, -1}, {1, -1},
    {-1, 0},  {0, 0},  {1, 0},
    {-1, 1},  {0, 1},  {1, 1},
}};

static bool bit(uint16_t code, int i) {
    return (code & (1u << i)) != 0;
}

static int neighborCount(uint16_t code) {
    int count = 0;
    for (int i = 0; i < 9; ++i) {
        if (i != kCenter && bit(code, i)) {
            ++count;
        }
    }
    return count;
}

static int foregroundCount(uint16_t code) {
    int count = 0;
    for (int i = 0; i < 9; ++i) {
        if (bit(code, i)) {
            ++count;
        }
    }
    return count;
}

cv::Mat loadBinaryImage(const std::string& path, int thresholdValue) {
    cv::Mat gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        throw std::runtime_error("Cannot read image: " + path);
    }

    cv::Mat binary;
    if (thresholdValue < 0) {
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    } else {
        cv::threshold(gray, binary, thresholdValue, 255, cv::THRESH_BINARY);
    }
    return binary;
}

uint16_t neighborhoodCode3x3(const cv::Mat& binary, int x, int y) {
    uint16_t code = 0;
    for (int i = 0; i < 9; ++i) {
        int nx = x + kOffsets[i].first;
        int ny = y + kOffsets[i].second;
        if (0 <= nx && nx < binary.cols && 0 <= ny && ny < binary.rows &&
            binary.at<uchar>(ny, nx) > 0) {
            code |= static_cast<uint16_t>(1u << i);
        }
    }
    return code;
}

cv::Mat applyLutOperator(const cv::Mat& binary, const std::array<uchar, 512>& lut) {
    cv::Mat out(binary.size(), CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            out.at<uchar>(y, x) = lut[neighborhoodCode3x3(binary, x, y)];
        }
    }
    return out;
}

bool matches(const cv::Mat& binary, int x, int y, const Kernel3x3& kernel) {
    for (int i = 0; i < 9; ++i) {
        if (kernel.cells[i] == Cell::DontCare) {
            continue;
        }
        int nx = x + kOffsets[i].first;
        int ny = y + kOffsets[i].second;
        bool value = 0 <= nx && nx < binary.cols && 0 <= ny && ny < binary.rows &&
                     binary.at<uchar>(ny, nx) > 0;
        if ((kernel.cells[i] == Cell::One) != value) {
            return false;
        }
    }
    return true;
}

cv::Mat hitOrMiss(const cv::Mat& binary, const Kernel3x3& kernel) {
    cv::Mat out(binary.size(), CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            if (matches(binary, x, y, kernel)) {
                out.at<uchar>(y, x) = 255;
            }
        }
    }
    return out;
}

static int localComponents8(uint16_t code, bool includeCenter) {
    std::array<bool, 9> seen{};
    int components = 0;
    for (int i = 0; i < 9; ++i) {
        if ((!includeCenter && i == kCenter) || !bit(code, i) || seen[i]) {
            continue;
        }
        ++components;
        std::queue<int> q;
        q.push(i);
        seen[i] = true;
        while (!q.empty()) {
            int cur = q.front();
            q.pop();
            for (int j = 0; j < 9; ++j) {
                if ((!includeCenter && j == kCenter) || !bit(code, j) || seen[j]) {
                    continue;
                }
                int dx = std::abs(kOffsets[cur].first - kOffsets[j].first);
                int dy = std::abs(kOffsets[cur].second - kOffsets[j].second);
                if (dx <= 1 && dy <= 1) {
                    seen[j] = true;
                    q.push(j);
                }
            }
        }
    }
    return components;
}

std::array<uchar, 512> buildCleanLut() {
    std::array<uchar, 512> lut{};
    for (uint16_t code = 0; code < 512; ++code) {
        bool keep = bit(code, kCenter) && neighborCount(code) > 0;
        lut[code] = keep ? 255 : 0;
    }
    return lut;
}

std::array<uchar, 512> buildFillLut() {
    std::array<uchar, 512> lut{};
    for (uint16_t code = 0; code < 512; ++code) {
        bool center = bit(code, kCenter);
        bool four = bit(code, 1) && bit(code, 3) && bit(code, 5) && bit(code, 7);
        lut[code] = (center || four) ? 255 : 0;
    }
    return lut;
}

std::array<uchar, 512> buildMajorityLut() {
    std::array<uchar, 512> lut{};
    for (uint16_t code = 0; code < 512; ++code) {
        lut[code] = foregroundCount(code) >= 5 ? 255 : 0;
    }
    return lut;
}

std::array<uchar, 512> buildEndpointsLut() {
    std::array<uchar, 512> lut{};
    for (uint16_t code = 0; code < 512; ++code) {
        lut[code] = bit(code, kCenter) && neighborCount(code) == 1 ? 255 : 0;
    }
    return lut;
}

std::array<uchar, 512> buildSpurLut() {
    std::array<uchar, 512> lut{};
    for (uint16_t code = 0; code < 512; ++code) {
        bool remove = bit(code, kCenter) && neighborCount(code) == 1;
        lut[code] = bit(code, kCenter) && !remove ? 255 : 0;
    }
    return lut;
}

std::array<uchar, 512> buildBridgeLut() {
    std::array<uchar, 512> lut{};
    for (uint16_t code = 0; code < 512; ++code) {
        if (bit(code, kCenter)) {
            lut[code] = 255;
            continue;
        }
        int before = localComponents8(code, false);
        int after = localComponents8(static_cast<uint16_t>(code | (1u << kCenter)), true);
        lut[code] = before >= 2 && after < before ? 255 : 0;
    }
    return lut;
}

std::array<uchar, 512> buildHbreakLut() {
    std::array<uchar, 512> lut{};
    for (uint16_t code = 0; code < 512; ++code) {
        bool horizontalBands = bit(code, 0) && bit(code, 1) && bit(code, 2) &&
                               bit(code, 6) && bit(code, 7) && bit(code, 8) &&
                               !bit(code, 3) && !bit(code, 5);
        bool verticalBands = bit(code, 0) && bit(code, 3) && bit(code, 6) &&
                             bit(code, 2) && bit(code, 5) && bit(code, 8) &&
                             !bit(code, 1) && !bit(code, 7);
        bool remove = bit(code, kCenter) && (horizontalBands || verticalBands);
        lut[code] = bit(code, kCenter) && !remove ? 255 : 0;
    }
    return lut;
}

static std::array<uchar, 512> buildLut(const std::string& op) {
    if (op == "clean") return buildCleanLut();
    if (op == "fill") return buildFillLut();
    if (op == "majority") return buildMajorityLut();
    if (op == "endpoints") return buildEndpointsLut();
    if (op == "spur") return buildSpurLut();
    if (op == "bridge") return buildBridgeLut();
    if (op == "hbreak") return buildHbreakLut();
    throw std::runtime_error("Unknown LUT operation: " + op);
}

static cv::Mat applyIterations(const cv::Mat& binary, const std::string& op, int iterations) {
    cv::Mat cur = binary.clone();
    auto lut = buildLut(op);
    for (int i = 0; i < std::max(1, iterations); ++i) {
        cur = applyLutOperator(cur, lut);
    }
    return cur;
}

static cv::Mat diffMap(const cv::Mat& before, const cv::Mat& after) {
    cv::Mat out(before.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    for (int y = 0; y < before.rows; ++y) {
        for (int x = 0; x < before.cols; ++x) {
            bool b = before.at<uchar>(y, x) > 0;
            bool a = after.at<uchar>(y, x) > 0;
            if (b && a) out.at<cv::Vec3b>(y, x) = {0, 0, 0};
            else if (!b && a) out.at<cv::Vec3b>(y, x) = {40, 170, 40};
            else if (b && !a) out.at<cv::Vec3b>(y, x) = {30, 30, 220};
        }
    }
    return out;
}

static int countComponents8(const cv::Mat& binary) {
    cv::Mat labels;
    int labelsCount = cv::connectedComponents(binary, labels, 8, CV_32S);
    return std::max(0, labelsCount - 1);
}

static int countPixels(const cv::Mat& binary) {
    return cv::countNonZero(binary);
}

static int countChanged(const cv::Mat& before, const cv::Mat& after) {
    cv::Mat diff;
    cv::compare(before, after, diff, cv::CMP_NE);
    return cv::countNonZero(diff);
}

static int countAdded(const cv::Mat& before, const cv::Mat& after) {
    int n = 0;
    for (int y = 0; y < before.rows; ++y) {
        for (int x = 0; x < before.cols; ++x) {
            n += before.at<uchar>(y, x) == 0 && after.at<uchar>(y, x) > 0;
        }
    }
    return n;
}

static int countRemoved(const cv::Mat& before, const cv::Mat& after) {
    int n = 0;
    for (int y = 0; y < before.rows; ++y) {
        for (int x = 0; x < before.cols; ++x) {
            n += before.at<uchar>(y, x) > 0 && after.at<uchar>(y, x) == 0;
        }
    }
    return n;
}

static Metrics metrics(const cv::Mat& image, const cv::Mat* before = nullptr) {
    Metrics m;
    m.pixels = countPixels(image);
    m.components8 = countComponents8(image);
    m.endpoints = countPixels(applyLutOperator(image, buildEndpointsLut()));
    if (before) {
        m.changed = countChanged(*before, image);
        m.added = countAdded(*before, image);
        m.removed = countRemoved(*before, image);
    }
    return m;
}

static cv::Mat scaledBinary(const cv::Mat& binary, int scale) {
    cv::Mat gray;
    cv::resize(binary, gray, cv::Size(), scale, scale, cv::INTER_NEAREST);
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

static cv::Mat scaledImage(const cv::Mat& image, int scale) {
    cv::Mat out;
    cv::resize(image, out, cv::Size(), scale, scale, cv::INTER_NEAREST);
    return out;
}

static cv::Mat panel(const cv::Mat& image, const std::string& label, int width = 220, int height = 220) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    if (resized.channels() == 1) {
        cv::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);
    }
    cv::Mat out(height + 34, width, CV_8UC3, cv::Scalar(245, 245, 245));
    resized.copyTo(out(cv::Rect(0, 34, width, height)));
    cv::putText(out, label, {10, 23}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {20, 20, 20}, 1, cv::LINE_AA);
    return out;
}

static cv::Mat hconcatPanels(const std::vector<cv::Mat>& panels) {
    cv::Mat out;
    cv::hconcat(panels, out);
    return out;
}

static void saveImage(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Cannot write image: " + path.string());
    }
}

static cv::Mat blank(int rows, int cols) {
    return cv::Mat(rows, cols, CV_8UC1, cv::Scalar(0));
}

static void writeLutCsv(const fs::path& path, const std::string& op) {
    auto lut = buildLut(op);
    std::ofstream f(path);
    f << "code,pattern,output_center\n";
    for (int code = 0; code < 512; ++code) {
        std::string pattern;
        for (int i = 0; i < 9; ++i) {
            pattern += bit(static_cast<uint16_t>(code), i) ? '1' : '0';
            if (i != 8) pattern += (i % 3 == 2) ? '/' : ' ';
        }
        f << code << ",\"" << pattern << "\"," << (lut[code] > 0 ? 1 : 0) << "\n";
    }
}

static void writeAllLutCsv(const fs::path& path) {
    std::vector<std::string> ops = {"clean", "fill", "majority", "endpoints", "spur", "bridge", "hbreak"};
    std::ofstream f(path);
    f << "operator,changed_configurations,ones_in_lut\n";
    for (const auto& op : ops) {
        auto lut = buildLut(op);
        int changed = 0;
        int ones = 0;
        for (int code = 0; code < 512; ++code) {
            bool center = bit(static_cast<uint16_t>(code), kCenter);
            bool out = lut[code] > 0;
            changed += center != out;
            ones += out;
        }
        f << op << "," << changed << "," << ones << "\n";
    }
}

static void putPixel(cv::Mat& img, int x, int y) {
    if (0 <= x && x < img.cols && 0 <= y && y < img.rows) {
        img.at<uchar>(y, x) = 255;
    }
}

static cv::Mat makeHitMissScene() {
    cv::Mat img = blank(50, 70);
    cv::line(img, {6, 10}, {28, 10}, 255, 1);
    cv::line(img, {45, 5}, {45, 28}, 255, 1);
    cv::line(img, {10, 35}, {25, 35}, 255, 1);
    cv::line(img, {25, 35}, {25, 45}, 255, 1);
    cv::line(img, {50, 38}, {62, 38}, 255, 1);
    putPixel(img, 50, 37);
    putPixel(img, 50, 39);
    return img;
}

static Kernel3x3 rightEndpointKernel() {
    return {{{Cell::Zero, Cell::Zero, Cell::Zero,
              Cell::One,  Cell::One,  Cell::Zero,
              Cell::Zero, Cell::Zero, Cell::Zero}}};
}

static cv::Mat makeBridgeScene() {
    cv::Mat img = blank(60, 90);
    cv::line(img, {6, 12}, {28, 12}, 255, 1);
    cv::line(img, {30, 12}, {55, 12}, 255, 1);
    cv::line(img, {10, 45}, {25, 30}, 255, 1);
    cv::line(img, {27, 28}, {45, 10}, 255, 1);
    cv::line(img, {60, 42}, {76, 42}, 255, 1);
    putPixel(img, 78, 41);
    return img;
}

static cv::Mat makeHbreakScene() {
    cv::Mat img = blank(35, 70);
    int x0 = 10;
    int y0 = 8;
    for (int x = x0; x < x0 + 9; ++x) {
        putPixel(img, x, y0);
        putPixel(img, x, y0 + 2);
    }
    putPixel(img, x0 + 4, y0 + 1);
    x0 = 43;
    y0 = 6;
    for (int y = y0; y < y0 + 9; ++y) {
        putPixel(img, x0, y);
        putPixel(img, x0 + 2, y);
    }
    putPixel(img, x0 + 1, y0 + 4);
    return img;
}

static cv::Mat makeSkeletonScene() {
    cv::Mat img = blank(70, 100);
    cv::line(img, {10, 35}, {78, 35}, 255, 1);
    cv::line(img, {40, 35}, {40, 12}, 255, 1);
    cv::line(img, {62, 35}, {78, 20}, 255, 1);
    cv::line(img, {25, 35}, {18, 48}, 255, 1);
    cv::line(img, {70, 35}, {88, 50}, 255, 1);
    return img;
}

struct GuiState {
    cv::Mat canvas;
    cv::Mat result;
    cv::Mat diff;
    std::vector<std::string> operations = {"clean", "fill", "majority", "endpoints",
                                           "spur", "bridge", "hbreak", "hitmiss"};
    int operationIndex = 5;
    int iterations = 1;
    int brush = 1;
    int scale = 8;
    int panelWidth = 0;
    int panelHeight = 0;
    bool drawing = false;
    bool erasing = false;
    bool hasResult = false;
};

static std::string currentOperation(const GuiState& state) {
    return state.operations[state.operationIndex];
}

static cv::Mat applyOperation(const cv::Mat& input, const std::string& op, int iterations) {
    if (op == "hitmiss") {
        return hitOrMiss(input, rightEndpointKernel());
    }
    return applyIterations(input, op, iterations);
}

static void drawOnCanvas(GuiState& state, int viewX, int viewY) {
    int imageY = viewY - 34;
    if (viewX < 0 || imageY < 0) {
        return;
    }
    int x = viewX / state.scale;
    int y = imageY / state.scale;
    if (x < 0 || x >= state.canvas.cols || y < 0 || y >= state.canvas.rows) {
        return;
    }
    int radius = std::max(0, state.brush - 1);
    cv::circle(state.canvas, {x, y}, radius, state.erasing ? 0 : 255, cv::FILLED, cv::LINE_8);
    state.hasResult = false;
}

static void onGuiMouse(int event, int x, int y, int flags, void* userdata) {
    auto* state = static_cast<GuiState*>(userdata);
    if (!state) {
        return;
    }

    bool leftDown = (flags & cv::EVENT_FLAG_LBUTTON) != 0;
    bool rightDown = (flags & cv::EVENT_FLAG_RBUTTON) != 0;
    if (event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_RBUTTONDOWN) {
        state->drawing = true;
        state->erasing = event == cv::EVENT_RBUTTONDOWN;
        drawOnCanvas(*state, x, y);
    } else if (event == cv::EVENT_MOUSEMOVE && state->drawing && (leftDown || rightDown)) {
        state->erasing = rightDown;
        drawOnCanvas(*state, x, y);
    } else if (event == cv::EVENT_LBUTTONUP || event == cv::EVENT_RBUTTONUP) {
        state->drawing = false;
    }
}

static cv::Mat makeGuiView(const GuiState& state) {
    cv::Mat canvasPanel = panel(scaledBinary(state.canvas, state.scale), "canvas",
                                state.panelWidth, state.panelHeight);

    cv::Mat resultImage = state.hasResult ? state.result : cv::Mat(state.canvas.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat resultPanel = panel(scaledBinary(resultImage, state.scale), currentOperation(state),
                                state.panelWidth, state.panelHeight);

    cv::Mat diffImage = state.hasResult ? state.diff : cv::Mat(state.canvas.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat diffPanel = panel(scaledImage(diffImage, state.scale), "diff",
                              state.panelWidth, state.panelHeight);

    cv::Mat body = hconcatPanels({canvasPanel, resultPanel, diffPanel});
    cv::Mat out(body.rows + 42, body.cols, CV_8UC3, cv::Scalar(245, 245, 245));
    body.copyTo(out(cv::Rect(0, 0, body.cols, body.rows)));

    Metrics before = metrics(state.canvas);
    std::ostringstream status;
    status << "op=" << currentOperation(state)
           << " iter=" << state.iterations
           << " brush=" << state.brush
           << " pixels=" << before.pixels
           << " components8=" << before.components8;
    if (state.hasResult) {
        Metrics after = metrics(state.result, &state.canvas);
        status << " -> pixels=" << after.pixels
               << " components8=" << after.components8
               << " changed=" << after.changed
               << " added=" << after.added
               << " removed=" << after.removed;
    }
    cv::putText(out, status.str(), {10, body.rows + 27}, cv::FONT_HERSHEY_SIMPLEX,
                0.52, {20, 20, 20}, 1, cv::LINE_AA);
    return out;
}

static void printGuiHelp() {
    std::cout
        << "GUI controls:\n"
        << "  Left mouse: draw foreground pixels\n"
        << "  Right mouse: erase pixels\n"
        << "  1..8: select clean, fill, majority, endpoints, spur, bridge, hbreak, hitmiss\n"
        << "  Space/Enter: apply selected operation\n"
        << "  [ and ]: decrease/increase iterations\n"
        << "  - and +: decrease/increase brush size\n"
        << "  c: clear canvas\n"
        << "  e: load built-in experiment sketch\n"
        << "  u: copy result back to canvas\n"
        << "  s: save canvas/result/diff to results/gui_*.png\n"
        << "  q or Esc: quit\n";
}

static void saveGuiState(const GuiState& state) {
    fs::create_directories("results");
    saveImage("results/gui_canvas.png", scaledBinary(state.canvas, state.scale));
    if (state.hasResult) {
        saveImage("results/gui_result.png", scaledBinary(state.result, state.scale));
        saveImage("results/gui_diff.png", scaledImage(state.diff, state.scale));
    }
}

static int runGui() {
    GuiState state;
    state.canvas = blank(84, 44);
    state.result = state.canvas.clone();
    state.diff = cv::Mat(state.canvas.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    state.panelWidth = state.canvas.cols * state.scale;
    state.panelHeight = state.canvas.rows * state.scale;

    const std::string windowName = "MorphLab GUI";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(windowName, onGuiMouse, &state);
    printGuiHelp();

    while (true) {
        cv::imshow(windowName, makeGuiView(state));
        int key = cv::waitKey(30);
        if (key < 0) {
            continue;
        }
        key &= 0xff;
        if (key == 27 || key == 'q') {
            break;
        } else if ('1' <= key && key <= '8') {
            state.operationIndex = key - '1';
            state.hasResult = false;
        } else if (key == ' ' || key == '\n' || key == '\r') {
            state.result = applyOperation(state.canvas, currentOperation(state), state.iterations);
            state.diff = diffMap(state.canvas, state.result);
            state.hasResult = true;
        } else if (key == '[') {
            state.iterations = std::max(1, state.iterations - 1);
            state.hasResult = false;
        } else if (key == ']') {
            state.iterations = std::min(50, state.iterations + 1);
            state.hasResult = false;
        } else if (key == '-' || key == '_') {
            state.brush = std::max(1, state.brush - 1);
        } else if (key == '+' || key == '=') {
            state.brush = std::min(8, state.brush + 1);
        } else if (key == 'c') {
            state.canvas.setTo(0);
            state.hasResult = false;
        } else if (key == 'e') {
            cv::resize(makeBridgeScene(), state.canvas, state.canvas.size(), 0, 0, cv::INTER_NEAREST);
            state.operationIndex = 5;
            state.hasResult = false;
        } else if (key == 'u' && state.hasResult) {
            state.canvas = state.result.clone();
            state.hasResult = false;
        } else if (key == 's') {
            saveGuiState(state);
            std::cout << "Saved GUI images to results/gui_*.png\n";
        }
    }

    cv::destroyWindow(windowName);
    return 0;
}

static void runExperiments(const fs::path& outDir) {
    fs::create_directories(outDir);

    std::ofstream report(outDir / "experiment_summary.md");
    report << "# Результаты экспериментов\n\n";

    cv::Mat hmInput = makeHitMissScene();
    cv::Mat hmOutput = hitOrMiss(hmInput, rightEndpointKernel());
    saveImage(outDir / "exp1_hitmiss_input.png", scaledBinary(hmInput, 8));
    saveImage(outDir / "exp1_hitmiss_matches.png", scaledBinary(hmOutput, 8));
    saveImage(outDir / "exp1_hitmiss_montage.png",
              hconcatPanels({panel(scaledBinary(hmInput, 8), "input"),
                             panel(scaledBinary(hmOutput, 8), "matches")}));
    report << "## Эксперимент 1: hit-or-miss\n\n";
    report << "Найдено совпадений: " << countPixels(hmOutput)
           << ". Использован шаблон правого конца горизонтальной линии `000/110/000`.\n\n";
    report << "![hit-or-miss](exp1_hitmiss_montage.png)\n\n";

    writeLutCsv(outDir / "exp2_lut_clean.csv", "clean");
    writeLutCsv(outDir / "exp2_lut_bridge.csv", "bridge");
    writeLutCsv(outDir / "exp2_lut_hbreak.csv", "hbreak");
    writeAllLutCsv(outDir / "exp2_lut_summary.csv");
    report << "## Эксперимент 2: полный перебор 512 окрестностей\n\n";
    report << "| Оператор | Изменяемых конфигураций | Единиц в LUT |\n";
    report << "|---|---:|---:|\n";
    std::vector<std::string> ops = {"clean", "fill", "majority", "endpoints", "spur", "bridge", "hbreak"};
    for (const auto& op : ops) {
        auto lut = buildLut(op);
        int changed = 0;
        int ones = 0;
        for (int code = 0; code < 512; ++code) {
            bool center = bit(static_cast<uint16_t>(code), kCenter);
            bool out = lut[code] > 0;
            changed += center != out;
            ones += out;
        }
        report << "|" << op << "|" << changed << "|" << ones << "|\n";
    }
    report << "\nCSV: `exp2_lut_clean.csv`, `exp2_lut_bridge.csv`, `exp2_lut_hbreak.csv`, `exp2_lut_summary.csv`.\n\n";

    cv::Mat bridgeInput = makeBridgeScene();
    cv::Mat bridgeOutput = applyIterations(bridgeInput, "bridge", 1);
    Metrics bmBefore = metrics(bridgeInput);
    Metrics bmAfter = metrics(bridgeOutput, &bridgeInput);
    cv::Mat bridgeDiff = scaledImage(diffMap(bridgeInput, bridgeOutput), 8);
    saveImage(outDir / "exp3_bridge_montage.png",
              hconcatPanels({panel(scaledBinary(bridgeInput, 8), "before"),
                             panel(scaledBinary(bridgeOutput, 8), "after"),
                             panel(bridgeDiff, "diff")}));
    report << "## Эксперимент 3: bridge\n\n";
    report << "| Компонент до | Компонент после | Добавлено пикселей | Изменено пикселей |\n";
    report << "|---:|---:|---:|---:|\n";
    report << "|" << bmBefore.components8 << "|" << bmAfter.components8 << "|"
           << bmAfter.added << "|" << bmAfter.changed << "|\n\n";
    report << "![bridge](exp3_bridge_montage.png)\n\n";

    cv::Mat hbInput = makeHbreakScene();
    cv::Mat hbOutput = applyIterations(hbInput, "hbreak", 1);
    Metrics hbAfter = metrics(hbOutput, &hbInput);
    cv::Mat hbDiff = scaledImage(diffMap(hbInput, hbOutput), 10);
    saveImage(outDir / "exp4_hbreak_montage.png",
              hconcatPanels({panel(scaledBinary(hbInput, 10), "before"),
                             panel(scaledBinary(hbOutput, 10), "after"),
                             panel(hbDiff, "diff")}));
    report << "## Эксперимент 4: hbreak\n\n";
    report << "Удалено пикселей: " << hbAfter.removed
           << ". Сработали горизонтальная и вертикальная версии H-шаблона.\n\n";
    report << "![hbreak](exp4_hbreak_montage.png)\n\n";

    cv::Mat skel = makeSkeletonScene();
    cv::Mat endpoints = applyIterations(skel, "endpoints", 1);
    cv::Mat spur1 = applyIterations(skel, "spur", 1);
    cv::Mat spur3 = applyIterations(skel, "spur", 3);
    cv::Mat spur5 = applyIterations(skel, "spur", 5);
    saveImage(outDir / "exp5_spur_montage.png",
              hconcatPanels({panel(scaledBinary(skel, 7), "input"),
                             panel(scaledBinary(endpoints, 7), "endpoints"),
                             panel(scaledBinary(spur1, 7), "spur 1"),
                             panel(scaledBinary(spur3, 7), "spur 3"),
                             panel(scaledBinary(spur5, 7), "spur 5")}));
    report << "## Эксперимент 5: endpoints и spur\n\n";
    report << "| Вариант | Пикселей объекта | Конечных точек | Компонент 8-связности |\n";
    report << "|---|---:|---:|---:|\n";
    for (const auto& row : std::vector<std::pair<std::string, cv::Mat>>{
             {"input", skel}, {"spur 1", spur1}, {"spur 3", spur3}, {"spur 5", spur5}}) {
        Metrics m = metrics(row.second);
        report << "|" << row.first << "|" << m.pixels << "|" << m.endpoints << "|"
               << m.components8 << "|\n";
    }
    report << "\n![spur](exp5_spur_montage.png)\n";
}

static std::map<std::string, std::string> parseArgs(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            throw std::runtime_error("Unexpected argument: " + key);
        }
        if (key == "--run-experiments" || key == "--gui") {
            args[key] = "1";
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + key);
        }
        args[key] = argv[++i];
    }
    return args;
}

static void usage() {
    std::cerr
        << "Usage:\n"
        << "  morphlab --input input.png --operation bridge --iterations 1 --output result.png\n"
        << "  morphlab --gui\n"
        << "  morphlab --run-experiments --output-dir results\n"
        << "Operations: clean, fill, majority, endpoints, spur, bridge, hbreak, hitmiss\n";
}

int main(int argc, char** argv) {
    try {
        auto args = parseArgs(argc, argv);
        if (args.count("--gui")) {
            return runGui();
        }

        if (args.count("--run-experiments")) {
            fs::path outDir = args.count("--output-dir") ? args["--output-dir"] : "results";
            runExperiments(outDir);
            std::cout << "Experiments written to " << outDir << "\n";
            return 0;
        }

        if (!args.count("--input") || !args.count("--operation") || !args.count("--output")) {
            usage();
            return 2;
        }

        int threshold = args.count("--threshold") ? std::stoi(args["--threshold"]) : -1;
        int iterations = args.count("--iterations") ? std::stoi(args["--iterations"]) : 1;
        cv::Mat input = loadBinaryImage(args["--input"], threshold);

        cv::Mat output;
        std::string op = args["--operation"];
        if (op == "hitmiss") {
            output = hitOrMiss(input, rightEndpointKernel());
        } else {
            output = applyIterations(input, op, iterations);
        }

        saveImage(args["--output"], output);
        if (args.count("--diff")) {
            saveImage(args["--diff"], diffMap(input, output));
        }

        Metrics before = metrics(input);
        Metrics after = metrics(output, &input);
        std::cout << "pixels_before=" << before.pixels
                  << " pixels_after=" << after.pixels
                  << " components8_before=" << before.components8
                  << " components8_after=" << after.components8
                  << " changed=" << after.changed
                  << " added=" << after.added
                  << " removed=" << after.removed << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        usage();
        return 1;
    }
}
