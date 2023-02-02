#include "LoopClosure.h"

void changeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

void mvo::wait()
{
    std::cout << std::endl << "Press enter to continue" << std::endl;
    getchar();
}

void mvo::LoopDetectCompute(const cv::Mat& img, std::vector<std::vector<cv::Mat>>& globaldesc, OrbDatabase& db)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cv::Mat mask;
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    DBoW2::QueryResults ret;
    orb->detectAndCompute(img, mask, kps, desc);
    globaldesc.push_back(std::vector<cv::Mat>());
    changeStructure(desc, globaldesc.back());
    db.add(globaldesc.back());
    db.query(globaldesc.back(), ret, 4);
    std::cout << "Searching for Image: " << ret << std::endl;    
}

void mvo::testDatabase(const std::vector<std::vector<cv::Mat>> &globaldesc, std::vector<mvo::MapData> map, OrbDatabase& db)
{
    std::cout << "Database information: " << std::endl << db << std::endl;

    // and query the database
    std::cout << "Querying the database: " << std::endl;

    int a = map.size();
    DBoW2::QueryResults ret;
    db.query(globaldesc.back(), ret, 4);
    // for(int i = 0; i < a; i++)
    // {
    //   db.query(globaldesc[i], ret, 4);

    //   // ret[0] is always the same image in this case, because we added it to the 
    //   // database. ret[1] is the second best match.

    //   // std::cout << "Searching for Image " << i << ". " << ret << std::endl;
    // }

    std::cout << std::endl;
    std::cout << "... done!" << std::endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    // std::cout << "Saving database..." << std::endl;
    // db.save("KITTI_00_phphww_db.yml.gz");
}


void mvo::VocCreation(const std::vector<std::vector<cv::Mat>>& features)
{
  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

  // branching factor and depth levels 
  const int k = 10;
  const int L = 4;
  const DBoW2::WeightingType weight = DBoW2::TF_IDF;
  const DBoW2::ScoringType scoring = DBoW2::L1_NORM;

  OrbVocabulary voc(k, L, weight, scoring);
  // BriefVocabulary voc(k, L, weight, scoring);
  std::cout << "Creating a small " << k << "^" << L << " vocabulary..." << std::endl;
  voc.create(features);
  std::cout << "... done!" << std::endl;


  // lets do something with this vocabulary
  std::cout << "Matching images against themselves (0 low, 1 high): " << std::endl;
  DBoW2::BowVector v1, v2;
  int NIMAGES = features.size();
  for(int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
    std::cout << "i: " << features.at(i).at(0) << std::endl;
    for(int j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);
      std::cout << "j: " << features.at(j).at(0) << std::endl;
      double score = voc.score(v1, v2);
      std::cout << "Image " << i << " vs Image " << j << ": " << score << std::endl;
    }
  }

  std::cout << "size: " << features.size() << std::endl;
  std::cout << "Vocabulary information: " << std::endl
  << voc << std::endl << std::endl;
  // save the vocabulary to disk
  std::cout << std::endl << "Saving vocabulary..." << std::endl;
  voc.save("KITTI_00_phphww_voc.yml.gz");
  std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
  std::cout << "Done : " << sec.count() << "secs" << std::endl;
//   OrbDatabase db(voc, false, 0);
//   TestDatabase(features, db);

}