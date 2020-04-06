#include <opencv2/opencv.hpp>
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#include <ros/ros.h>
#include <ros/package.h>
#include<iostream>
#include<vector>
#include <cmath>


class SuperPointNetwork{
  public: 
    SuperPointNetwork(){
      // Initialize a tensorflow session
      options.config.mutable_gpu_options()->set_allow_growth(true);
      status = NewSession(options, &session);    
      // prepare session
      TF_CHECK_OK(tensorflow::LoadSavedModel(sess_options, run_options, \
        graph_fn, {tensorflow::kSavedModelTagServe}, &bundle));
      graph_def = bundle.meta_graph_def;
    }
    ~SuperPointNetwork(){
      session-> Close();
    }
    std::unique_ptr<tensorflow::Session>&sess = bundle.session;
  
  private:
    tensorflow::Session* session;
    tensorflow::SessionOptions options = tensorflow::SessionOptions();
    tensorflow::Status status;
    const std::string& graph_fn = "/home/robin/work/src/Superpoint/models/sp_v6";
    tensorflow::SessionOptions sess_options;
    tensorflow::RunOptions run_options;
    tensorflow::MetaGraphDef graph_def;
    tensorflow::SavedModelBundle bundle;
};

class SuperPoint{
  
  public:
    void compute(cv::Mat& im, cv::Mat & desc, std::vector<cv::KeyPoint>&keyPoints,std::unique_ptr<tensorflow::Session>& sess);
    SuperPoint(int keyPtNum){
      this->keyPtNum = keyPtNum;
      rows=480;
      cols=640;
    }
  
  private:
    int rows;
    int cols;
    int keyPtNum; 
    void imageProcessing(cv::Mat& img);
    void cvmatToTensor(cv::Mat img,tensorflow::Tensor* output_tensor,int input_rows,int input_cols);
    void extract(int rows,int cols,int keyPtNum,float*keyData,Eigen::TensorMap<Eigen::Tensor<float, 4, 1, long>, 16>&descriptorTensor2,std::vector<cv::KeyPoint>&keyPoints,cv::Mat&desc);
    std::vector<int> sortIndexes(float* v,int size);
};

void SuperPoint::cvmatToTensor(cv::Mat img,tensorflow::Tensor* output_tensor,int input_rows,int input_cols){
    img.convertTo(img,CV_32FC1);
    img= img/255;
    float *p = output_tensor->flat<float>().data();
    cv::Mat tempMat(input_rows, input_cols, CV_32FC1, p);
    img.convertTo(tempMat,CV_32FC1);
}

/////argsort
std::vector<int> SuperPoint::sortIndexes(float* v,int size){

  // initialize original index locations
  std::vector<int> idx(size);
  std::iota(idx.begin(), idx.end(),0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  std::stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
  return idx;
}

void SuperPoint::imageProcessing(cv::Mat&im){
  cvtColor(im, im, CV_RGB2GRAY);
  resize(im,im,cv::Size(cols,rows));
}

void SuperPoint::extract(int rows,int cols,int keyPtNum,float*keyData,Eigen::TensorMap<Eigen::Tensor<float, 4, 1, long>, 16>&descriptorTensor2,std::vector<cv::KeyPoint>&keyPoints,cv::Mat&desc){
  std::vector<int> index = sortIndexes(keyData,rows*cols);
  for(int i = 0 ; i < keyPtNum ; i ++){
    int ii = index[i];
    int row_ = std::floor(ii/cols);
    int col_ = ii%cols;
    keyPoints.push_back(cv::KeyPoint(col_, row_, 1));
    for(int j = 0 ; j < 256 ; j ++){
      desc.at<float>(i,j)= descriptorTensor2(0,row_,col_,j);
    }
  }
}
void SuperPoint::compute(cv::Mat&im, cv::Mat &desc, std::vector<cv::KeyPoint>&keyPoints,std::unique_ptr<tensorflow::Session>&sess){
    
    // process image and turn into tensor
    imageProcessing(im);
    tensorflow::Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,rows, cols,1}));
    cvmatToTensor(im,&x,rows,cols);

    // input the tensor into the neural network
    std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs;
    inputs.push_back(std::pair<std::string, tensorflow::Tensor>("superpoint/image", x));
    std::vector<tensorflow::Tensor> outputTensor;
    TF_CHECK_OK(sess -> Run(inputs, {"superpoint/prob_nms:0", "superpoint/descriptors:0"}, {}, &outputTensor));
    
    // extract the descriptor and keypoints from output tensor
    auto keyTensor = outputTensor[0].tensor<float, 3>();
    float* keyData = keyTensor.data();
    auto descriptorTensor = outputTensor[1].tensor<float, 4>();
    
    extract(rows,cols,keyPtNum,keyData,descriptorTensor,keyPoints,desc);
}
int main(int argc, char* argv[]) {
  ros::init(argc, argv, "tensorflow_test_with_ros");
  
  cv::Mat im = imread("/home/robin/SuperPointPretrainCpp_TensorFlow/data/img2.jpg",cv::IMREAD_COLOR);
  cv::Mat im2 = imread("/home/robin/SuperPointPretrainCpp_TensorFlow/data/img1.jpg",cv::IMREAD_COLOR);
  int keyPtNum = 2000;
 
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
SuperPointNetwork spn;
SuperPoint sp1(keyPtNum);
std::vector<cv::KeyPoint> keyPoints;
cv::Mat desc = cv::Mat(keyPtNum, 256, CV_32F);
sp1.compute(im,desc,keyPoints,spn.sess);

SuperPoint sp2(keyPtNum);
std::vector<cv::KeyPoint> keyPoints2;
cv::Mat desc2 = cv::Mat(keyPtNum, 256, CV_32F);
sp2.compute(im2,desc2,keyPoints2,spn.sess);


///////////////////////////////////////Key Points Test/////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Mat output;
//cv::drawKeypoints(im, keyPoints, output, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
//cv::imshow("Output", output);
//cv::waitKey(0);

  
///////////////////////////////////Descriptors Test ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  std::vector< std::vector<cv::DMatch> > knn_matches;
  matcher->knnMatch( desc, desc2, knn_matches, 2);
  //-- Draw matches
  const float ratio_thresh = 0.7;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
  
  cv::Mat img_matches;
  drawMatches(im, keyPoints,im2,keyPoints2, good_matches, img_matches, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
  std::vector< cv::Point2f > obj;
  std::vector< cv::Point2f > scene;
  for( int i = 0; i < good_matches.size(); i++){
  //-- Get the keypoints from the good matches
    obj.push_back(keyPoints[good_matches[i].queryIdx ].pt);
    scene.push_back(keyPoints2[good_matches[i].trainIdx ].pt);
  }

  cv::Mat H = findHomography( obj, scene, CV_RANSAC);
  cv::Mat result;
  cv::warpPerspective(im,result,H,cv::Size(im.cols+im2.cols,im.rows));
  cv::Mat half(result,cv::Rect(0,0,im2.cols,im2.rows));
  im2.copyTo(half);

  //imshow("Good Matches", img_matches );
  cv::imshow( "Result", result );
  cv::waitKey();
  //session->Close();
  return 0;
}
