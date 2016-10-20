
#include "mnist.h"


#include<opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include <string>
#include <iostream>

using namespace std;
using namespace cv;
//using namespace cv::ml;

//计时器
double cost_time;
clock_t start_time;
clock_t end_time;

//测试item个数
int testNum = 10000;

//大端存储转换为小端存储
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;//右移位　与操作
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

Mat read_mnist_image(const string fileName) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    Mat DataMat;


    const char* charFileName = fileName.c_str();//string 转换为 char*
    ifstream file(charFileName,ios::binary);

    //ifstream file(fileName, ios::binary);
    if (file.is_open())
    {
        cout << "成功打开图像集 ... \n";

        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_rows, sizeof(n_rows));
        file.read((char*)&n_cols, sizeof(n_cols));
        //cout << magic_number << " " << number_of_images << " " << n_rows << " " << n_cols << endl;


        magic_number = reverseInt(magic_number);
        number_of_images = reverseInt(number_of_images);
        n_rows = reverseInt(n_rows);
        n_cols = reverseInt(n_cols);
        cout << "MAGIC NUMBER = " << magic_number
            << " ;NUMBER OF IMAGES = " << number_of_images
            << " ; NUMBER OF ROWS = " << n_rows
            << " ; NUMBER OF COLS = " << n_cols << endl;

        //-test-
        //number_of_images = testNum;
        //输出第一张和最后一张图，检测读取数据无误
        Mat s = Mat::zeros(n_rows, n_cols, CV_32FC1);
        Mat e = Mat::zeros(n_rows, n_cols, CV_32FC1);

        cout << "开始读取Image数据......\n";
        start_time = clock();
        DataMat = Mat::zeros(number_of_images, n_rows * n_cols, CV_32FC1);
        for (int i = 0; i < number_of_images; i++) {
            for (int j = 0; j < n_rows * n_cols; j++) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                float pixel_value = float((temp + 0.0) / 255.0);
                DataMat.at<float>(i, j) = pixel_value;

                //打印第一张和最后一张图像数据
                if (i == 0) {
                    s.at<float>(j / n_cols, j % n_cols) = pixel_value;
                }
                else if (i == number_of_images - 1) {
                    e.at<float>(j / n_cols, j % n_cols) = pixel_value;
                }
            }
        }
        end_time = clock();
        cost_time = (end_time - start_time) / CLOCKS_PER_SEC;
        cout << "读取Image数据完毕......" << cost_time << "s\n";

        namedWindow("first image",CV_WINDOW_NORMAL);
        imshow("first image", s);
        namedWindow("last image",CV_WINDOW_NORMAL);
        imshow("last image", e);
        waitKey(0);
        destroyAllWindows();
    }
    file.close();
    return DataMat;
}

Mat read_mnist_label(const string fileName) {
    int magic_number;
    int number_of_items;

    Mat LabelMat;

    const char* charFileName = fileName.c_str();
    ifstream file(charFileName, ios::binary);
    if (file.is_open())
    {
        cout << "成功打开Label集 ... \n";

        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_items, sizeof(number_of_items));
        magic_number = reverseInt(magic_number);
        number_of_items = reverseInt(number_of_items);

        cout << "MAGIC NUMBER = " << magic_number << "  ; NUMBER OF ITEMS = " << number_of_items << endl;

        //-test-
        //number_of_items = testNum;
        //记录第一个label和最后一个label
        unsigned int s = 0, e = 0;

        cout << "开始读取Label数据......\n";
        start_time = clock();
        LabelMat = Mat::zeros(number_of_items, 1, CV_32SC1);
        for (int i = 0; i < number_of_items; i++) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            LabelMat.at<unsigned int>(i, 0) = (unsigned int)temp;

            //打印第一个和最后一个label
            if (i == 0) s = (unsigned int)temp;
            else if (i == number_of_items - 1) e = (unsigned int)temp;
        }
        end_time = clock();
        cost_time = (end_time - start_time) / CLOCKS_PER_SEC;
        cout << "读取Label数据完毕......" << cost_time << "s\n";

        cout << "first label = " << s << endl;
        cout << "last label = " << e << endl;
    }
    file.close();
    return LabelMat;
}

/*
svm_type –
指定SVM的类型，下面是可能的取值：
CvSVM::C_SVC C类支持向量分类机。 n类分组  (n \geq 2)，允许用异常值惩罚因子C进行不完全分类。
CvSVM::NU_SVC \nu类支持向量分类机。n类似然不完全分类的分类器。参数为 \nu 取代C（其值在区间【0，1】中，nu越大，决策边界越平滑）。
CvSVM::ONE_CLASS 单分类器，所有的训练数据提取自同一个类里，然后SVM建立了一个分界线以分割该类在特征空间中所占区域和其它类在特征空间中所占区域。
CvSVM::EPS_SVR \epsilon类支持向量回归机。训练集中的特征向量和拟合出来的超平面的距离需要小于p。异常值惩罚因子C被采用。
CvSVM::NU_SVR \nu类支持向量回归机。 \nu 代替了 p。

可从 [LibSVM] 获取更多细节。

kernel_type –
SVM的内核类型，下面是可能的取值：
CvSVM::LINEAR 线性内核。没有任何向映射至高维空间，线性区分（或回归）在原始特征空间中被完成，这是最快的选择。K(x_i, x_j) = x_i^T x_j.
CvSVM::POLY 多项式内核： K(x_i, x_j) = (\gamma x_i^T x_j + coef0)^{degree}, \gamma > 0.
CvSVM::RBF 基于径向的函数，对于大多数情况都是一个较好的选择： K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}, \gamma > 0.
CvSVM::SIGMOID Sigmoid函数内核：K(x_i, x_j) = \tanh(\gamma x_i^T x_j + coef0).

degree – 内核函数（POLY）的参数degree。

gamma – 内核函数（POLY/ RBF/ SIGMOID）的参数\gamma。

coef0 – 内核函数（POLY/ SIGMOID）的参数coef0。

Cvalue – SVM类型（C_SVC/ EPS_SVR/ NU_SVR）的参数C。

nu – SVM类型（NU_SVC/ ONE_CLASS/ NU_SVR）的参数 \nu。

p – SVM类型（EPS_SVR）的参数 \epsilon。

class_weights – C_SVC中的可选权重，赋给指定的类，乘以C以后变成 class\_weights_i * C。所以这些权重影响不同类别的错误分类惩罚项。权重越大，某一类别的误分类数据的惩罚项就越大。

term_crit – SVM的迭代训练过程的中止条件，解决部分受约束二次最优问题。您可以指定的公差和/或最大迭代次数。

*/




string trainImage = "./mnist/train-images.idx3-ubyte";
string trainLabel = "./mnist/train-labels.idx1-ubyte";
string testImage = "./mnist/t10k-images.idx3-ubyte";
string testLabel = "./mnist/t10k-labels.idx1-ubyte";
//string testImage = "./mnist/train-images.idx3-ubyte";
//string testLabel = "./mnist/train-labels.idx1-ubyte";


//计时器
double cost_time_;
clock_t start_time_;
clock_t end_time_;

int main()
{

    //--------------------- 1. Set up training data ---------------------------------------
    Mat trainData;
    Mat labels;
    trainData = read_mnist_image(trainImage);
    labels = read_mnist_label(trainLabel);


    cout << trainData.rows << " " << trainData.cols << endl;
    cout << labels.rows << " " << labels.cols << endl;

    //------------------------ 2. Set up the support vector machines parameters --------------------

    SVM MY_SVM;

    cv::SVMParams params;
    params.svm_type = SVM::C_SVC;

    params.kernel_type = SVM::POLY;
    params.C = 10.0;
    params.gamma = 0.01;
    params.degree= 3;
    params.term_crit = cv::TermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);


    MY_SVM.get_params();
//    Ptr<SVM> svm = SVM::create();
//    svm->setType(SVM::C_SVC);
//    svm->setKernel(SVM::RBF);
//    //svm->setDegree(10.0);
//    svm->setGamma(0.01);
//    //svm->setCoef0(1.0);
//    svm->setC(10.0);
//    //svm->setNu(0.5);
//    //svm->setP(0.1);
//    svm->setTermCriteria(TermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON));

    //------------------------ 3. Train the svm ----------------------------------------------------
    cout << "Starting training process" << endl;
    start_time_ = clock();
    MY_SVM.train(trainData, labels,Mat(),Mat(),params);
    end_time_ = clock();
    cost_time_ = (end_time_ - start_time_) / CLOCKS_PER_SEC;
    cout << "Finished training process...cost " << cost_time_ << " seconds..." << endl;

    //------------------------ 4. save the svm ----------------------------------------------------
    MY_SVM.save("./mnist_dataset/mnist_svm.xml");
    cout << "save as ./mnist_dataset/mnist_svm.xml" << endl;


    //------------------------ 5. load the svm ----------------------------------------------------
    const char* model = "./mnist_dataset/mnist_svm.xml";
    cout << "开始导入SVM文件...\n";
    MY_SVM.load(model);
    //Ptr<SVM> svm1 = cv::StatModel::load<SVM>(model);
    cout << "成功导入SVM文件...\n";


    //------------------------ 6. read the test dataset -------------------------------------------
    cout << "开始导入测试数据...\n";
    Mat testData;
    Mat tLabel;
    testData = read_mnist_image(testImage);
    tLabel = read_mnist_label(testLabel);
    cout << "成功导入测试数据！！！\n";


    float count = 0;
    for (int i = 0; i < testData.rows; i++) {
        Mat sample = testData.row(i);
        float res = MY_SVM.predict(sample);
        res = std::abs(res - tLabel.at<unsigned int>(i, 0)) <= FLT_EPSILON ? 1.f : 0.f;
        count += res;
    }
    cout << "正确的识别个数 count = " << count << endl;
    cout << "错误率为..." << (10000 - count + 0.0) / 10000 * 100.0 << "%....\n";

    return 0;
}

