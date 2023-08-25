#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QString>
#include <QDebug>


int blurSize = 3;
int cannyLowThreshold = 35;
int cannyHighThreshold = 160;
int houghThreshold = 100;
int houghMinLineLength = 300;
int houghMaxLineGap = 10;


const int DEBUG_LEVEL = 2; // 1 - text debug; 2 - debug with result images; 3 - debug with filtered images; 4 - debug with sliders


void splitBook(const cv::Mat& inputImage, cv::Mat& leftPage, cv::Mat& rightPage) {
    cv::Mat gray, binarized, morphed, debugImage, blurred;

    cv::Mat image ;
    cv::resize(inputImage.clone(), image, cv::Size(),  0.2, 0.2);
    debugImage = image.clone();

    // 1. Бинаризация
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::medianBlur(gray, gray, 5);

    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::adaptiveThreshold(blurred, binarized, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 25, 4);
    if (DEBUG_LEVEL >= 2) {
        cv::imshow("Binarized Image", binarized);
    }

    // 3. Проекция
    cv::Mat projection;
    cv::reduce(binarized, projection, 0, cv::REDUCE_SUM, CV_32S);

    // 4. Поиск минимального значения в центральной области изображения
    int center = image.cols / 2;
    int window = image.cols * 0.2; // Проверяем 20% изображения вокруг центра

    double minVal;
    cv::Point minLoc;
    cv::Mat centerRegion = projection(cv::Rect(center - window / 2, 0, window, 1));
    cv::minMaxLoc(centerRegion, &minVal, nullptr, &minLoc);

    int splitPosition = center - window / 2 + minLoc.x;

    if (DEBUG_LEVEL >= 2) {
        cv::line(debugImage, cv::Point(splitPosition, 0), cv::Point(splitPosition, image.rows), cv::Scalar(0, 0, 255), 2);
        cv::imshow("Debug Image with Split Line", debugImage);
    }

    // 5. Разделение
    leftPage = image(cv::Rect(0, 0, splitPosition, image.rows));
    rightPage = image(cv::Rect(splitPosition, 0, image.cols - splitPosition, image.rows));

    if (DEBUG_LEVEL >= 2) {
        cv::imshow("Left Page", leftPage);
        cv::imshow("Right Page", rightPage);
        cv::waitKey(0);
    }
}



std::vector<cv::Point> findPageCorners(const cv::Mat& image) {
    cv::Mat gray, blurred, binarized, dilated, closed, inverted;

    // Преобработка изображения
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 5);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::adaptiveThreshold(blurred, binarized, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

    // Морфологические операции
    cv::dilate(binarized, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    cv::morphologyEx(dilated, closed, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));
    cv::bitwise_not(closed, inverted);
    cv::dilate(inverted, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8)));
    cv::erode(dilated, closed, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8)));

    // Нахождение контуров для создания маски
    std::vector<std::vector<cv::Point>> externalContours;
    cv::findContours(closed, externalContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat mask = cv::Mat::zeros(closed.size(), CV_8U);
    if (!externalContours.empty()) {
        int largestExternalContourIdx = std::distance(externalContours.begin(), std::max_element(externalContours.begin(), externalContours.end(), [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
            return cv::contourArea(c1) < cv::contourArea(c2);
        }));
        cv::drawContours(mask, externalContours, largestExternalContourIdx, cv::Scalar(255), -1);
    }

    // Нахождение контуров и фильтрация их с учетом маски
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    contours.erase(std::remove_if(contours.begin(), contours.end(),
                                  [&mask](const std::vector<cv::Point>& contour) {
                                      for (const auto& point : contour) {
                                          if (mask.at<uchar>(point) == 0) {
                                              return true;  // удалить контур
                                          }
                                      }
                                      return false;
                                  }),
                   contours.end());

    double minContourArea = 0.01 * image.cols * image.rows;
    contours.erase(std::remove_if(contours.begin(), contours.end(),
                                  [minContourArea](const std::vector<cv::Point>& contour) {
                                      return cv::contourArea(contour) < minContourArea;
                                  }),
                   contours.end());

    // Сортировка контуров по убыванию площади
    std::sort(contours.begin(), contours.end(),
              [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
                  return cv::contourArea(c1) > cv::contourArea(c2);
              });

    if (contours.empty()) {
        return {};
    }

    // Выбор наибольшего контура и его аппроксимация
    std::vector<cv::Point> largestContour = contours[0];

    // Применяем convexHull к наибольшему контуру
    std::vector<cv::Point> hull;
    cv::convexHull(largestContour, hull);

    std::vector<cv::Point> approx;
    cv::approxPolyDP(hull, approx, cv::arcLength(largestContour, true) * 0.02, true);

    if (DEBUG_LEVEL >= 2) {
        cv::Mat debugClosedImage = cv::Mat::zeros(closed.size(), CV_8UC3);
        std::vector<std::vector<cv::Point>> contoursToDraw = { largestContour };
        cv::drawContours(debugClosedImage, contoursToDraw, -1, cv::Scalar(0, 255, 0), 2);

        if (DEBUG_LEVEL >= 3) {
            cv::resize(binarized, binarized, cv::Size(), 0.1, 0.1);
            cv::resize(inverted, inverted, cv::Size(), 0.1, 0.1);
            cv::resize(closed, closed, cv::Size(), 0.1, 0.1);

            cv::imshow("Binarized Image", binarized);
            cv::imshow("Inverted Image", inverted);
            cv::imshow("Closed Image", closed);
        }

        cv::resize(debugClosedImage, debugClosedImage, cv::Size(), 0.1, 0.1);
        cv::imshow("Closed Image with Contours", debugClosedImage);
        cv::waitKey(0);
    }

    return approx;
}


cv::Vec4i findCenterBookLine(const cv::Mat &inputMat) {
    cv::Mat processedMat, debugMat, processedMatDebug, cannyOutput, blurred, binarized, gray, dilated, canny;
    inputMat.copyTo(debugMat);

    cv::cvtColor(inputMat, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 1, 1);
    //cv::adaptiveThreshold(blurred, processedMat, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

    cv::Canny(blurred, canny, cannyLowThreshold, cannyHighThreshold, 3);
    //processedMat = cannyOutput;

    // Морфологические операции
    cv::dilate(canny, dilated, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));
    cv::morphologyEx(dilated, processedMat, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 7)));

    cv::Mat sobel;
    cv::Sobel(processedMat, sobel, CV_8U, 1, 0, 15);
    cv::threshold(sobel, processedMat, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);


    if (DEBUG_LEVEL >= 2) {
        if (DEBUG_LEVEL >= 3) {
            cv::resize(processedMat, processedMatDebug, cv::Size(), 0.1, 0.1);
            cv::imshow("Canny", processedMatDebug);
            cv::waitKey(0);
        }
    }

    std::vector<cv::Vec4i> lines;
     cv::HoughLinesP(processedMat, lines, 1, CV_PI/180, houghThreshold, houghMinLineLength, houghMaxLineGap);
    //cv::HoughLinesP(processedMat, lines, 1, CV_PI/180, 150, 150, 1200);

    if (lines.empty()) {
        if (DEBUG_LEVEL >= 1) {
            std::cout << "No lines detected!" << std::endl;
        }
        return cv::Vec4i();  // Return an empty line if no lines are detected
    }

    double centerOfImage = inputMat.cols / 2.0;
    double minDistanceToCenter = std::numeric_limits<double>::max();
    int bestLineIndex = -1;

    for (size_t i = 0; i < lines.size(); ++i) {
        const auto &line = lines[i];
        if (std::abs(line[0] - line[2]) < std::abs(line[1] - line[3])) {
            double lineCenter = (line[0] + line[2]) / 2.0;
            double distanceToCenter = std::abs(lineCenter - centerOfImage);
            if (distanceToCenter < minDistanceToCenter) {
                minDistanceToCenter = distanceToCenter;
                bestLineIndex = i;
            }
        }
    }

    cv::Vec4i bestLine = (bestLineIndex != -1) ? lines[bestLineIndex] : cv::Vec4i();

    // If line is vertical or nearly vertical, we'll just extend it from top to bottom directly
    if (std::abs(bestLine[0] - bestLine[2]) < 1e-6) {
        bestLine[1] = 0;
        bestLine[3] = inputMat.rows - 1;
    } else {
        // Calculate the slope m of the line
        double m = static_cast<double>(bestLine[3] - bestLine[1]) / (bestLine[2] - bestLine[0]);
        double c = bestLine[1] - m * bestLine[0];

        int yTop = 0;
        int xTop = (yTop - c) / m;

        int yBottom = inputMat.rows - 1;
        int xBottom = (yBottom - c) / m;

        bestLine = cv::Vec4i(xTop, yTop, xBottom, yBottom);
    }

    if (DEBUG_LEVEL >= 2) {
        cv::line(debugMat, cv::Point(bestLine[0], bestLine[1]), cv::Point(bestLine[2], bestLine[3]), cv::Scalar(0, 0, 255), 10);
        cv::resize(debugMat, debugMat, cv::Size(), 0.1, 0.1);
        cv::imshow("Detected Center Line", debugMat);
        cv::waitKey(0);
    }

    return bestLine;
}


cv::Mat alignBookVertically(const cv::Mat &inputMat) {
    // Find the center line of the book.
    cv::Vec4i centerLine = findCenterBookLine(inputMat);

    // Compute the angle of this line with the vertical axis (y-axis).
    double angle = std::atan2(static_cast<double>(centerLine[3] - centerLine[1]),
                              static_cast<double>(centerLine[2] - centerLine[0]));

    // Convert the angle from radians to degrees.
    angle = angle * 180 / CV_PI;

    // Find the midpoint of the line which will be used as the rotation center.
    cv::Point2f center(centerLine[0] + (centerLine[2] - centerLine[0]) * 0.5,
                       centerLine[1] + (centerLine[3] - centerLine[1]) * 0.5);

    // Get the rotation matrix.
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);

    // Apply the rotation to the image.
    cv::Mat alignedMat;
    cv::warpAffine(inputMat, alignedMat, rotMat, inputMat.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

    return alignedMat;
}


void drawContoursOnImage(const cv::Mat& inputImage, const std::vector<cv::Point>& contours, cv::Mat& outputImage) {
    // Копируем изображение
    inputImage.copyTo(outputImage);

    if(contours.size() == 4) {
        // Рисуем контуры на изображении
        std::vector<std::vector<cv::Point>> contourVec = {contours};
        cv::drawContours(outputImage, contourVec, -1, cv::Scalar(0, 255, 0), 2); // Зеленый цвет для контуров

        // Рисуем углы (точки) контура
        for (const auto& pt : contours) {
            cv::circle(outputImage, pt, 5, cv::Scalar(255, 0, 0), -1); // Синий цвет для углов
        }
    }
}

/*
void imgAnalyze(cv::Mat &page){


    if (!page.empty()) {

        //cv::resize(page, page, cv::Size(), 0.2, 0.2);

        std::vector<cv::Point> corners = findPageCorners(page);

        cv::Mat outputImage;
        drawContoursOnImage(page, corners, outputImage);

        if (DEBUG_LEVEL >= 2) {
            cv::resize(outputImage, outputImage, cv::Size(), 0.1, 0.1);
            cv::imshow("Detected Page Contours", outputImage);
            cv::waitKey(0);
        }
    } else {
        std::cerr << "Error loading image!" << std::endl;
    }
}*/


// Создание функции обратного вызова
void onTrackbarChanged(int, void* userData) {
    cv::Mat* matPtr = static_cast<cv::Mat*>(userData);
    findCenterBookLine(*matPtr);
}


int main() {

    QString pathToFolder = "C:/Users/danoc/Documents/geometry/";
    for (int i = 1; i < 13; i++) {
        QString path = pathToFolder +  QString::number(i) + ".jpg";

        cv::Mat image = cv::imread(path.toLocal8Bit().constData());
        cv::Mat inputMat = image.clone();
        cv::Mat leftPage, rightPage;
        //splitBook(image, leftPage, rightPage);


        // Создание окна и трекбаров
           if (DEBUG_LEVEL >= 4) {
               //


               cv::namedWindow("Controls", cv::WINDOW_AUTOSIZE);

               cv::createTrackbar("Blur Size", "Controls", &blurSize, 25, onTrackbarChanged, &inputMat);
               cv::createTrackbar("Canny Low", "Controls", &cannyLowThreshold, 255, onTrackbarChanged, &inputMat);
               cv::createTrackbar("Canny High", "Controls", &cannyHighThreshold, 255, onTrackbarChanged, &inputMat);
               cv::createTrackbar("Hough Threshold", "Controls", &houghThreshold, 300, onTrackbarChanged, &inputMat);
               cv::createTrackbar("Hough Min Line Length", "Controls", &houghMinLineLength, 1500, onTrackbarChanged, &inputMat);
               cv::createTrackbar("Hough Max Line Gap", "Controls", &houghMaxLineGap, 1500, onTrackbarChanged, &inputMat);


                // Запуск главного цикла для работы трек баров
                while (true) {
                   findCenterBookLine(image); // Обновление изображения

                   char key = (char) cv::waitKey(10);
                   if (key == 27)  // Если нажата клавиша 'Esc', выходите из цикла
                       break;
                }
            } else {
               alignBookVertically(image);

           }
    }



    //cv::resize(image, image, cv::Size(), 0.1, 0.1);
    //alignBookVertically(image);

       //imgAnalyze(leftPage);
    //imgAnalyze(rightPage);



    return 0;
}





