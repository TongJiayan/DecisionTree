//
// Created by JoeyTong on 2021/5/14.
//

#define TRAIN_SIZE_PER_CLASS 40 //iris数据集每一类样本为50个，选择其中40个样本作为训练数据，其余作为测试数据
#define ATTR_SIZE 4
#define CLASS_NUM 3

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <cmath>
using namespace std;

struct Data{
    vector<int> attr;
    int label;
};

typedef pair<int, vector<int>> AttrValueItem; // (attrValue,(numClass0, numClass1, numClass2))
struct Attr{
    int attr;
    vector<AttrValueItem> valueAndNum;
};

struct Node{
    bool isLeafNode;                        // default = false
    int classLabel;                         // default = -1
    int attr;                               // 节点选择的用于判断的属性：0，1，2，3
    int attrValue;                          // 父节点的划分属性值
    vector<Node*> children;
    Node();
};

Node::Node() : isLeafNode(false), classLabel(-1), attr(-1), attrValue(-1),children(0){}

/* 读取数据文件存到totalData中，考虑到原始数据文件的有序性和后续层次划分的需求
 * 将数据按所属类划分到三个vector中，其中totalData[0] 对应第一类数据，依次类推 */
void readData(const string& filename, vector<vector<Data>>& totalData){
    ifstream fin(filename, ios::in);
    if(fin.fail()){
        cout<<"打开数据文件失败！"<<endl;
        exit(1);
    }
    string line;
    vector<Data> dataPerClass;
    int currentLabel = 0;
    while(getline(fin, line)){
        istringstream sin(line);
        string field;
        Data data;
        data.attr.clear();
        while (getline(sin, field, ',')){
            if(data.attr.size() < ATTR_SIZE)
                data.attr.push_back(atoi(field.c_str()));
            else
                data.label = atoi(field.c_str());
        }
        if(data.label!=currentLabel){           // 开始读取下一类
            totalData.push_back(dataPerClass);
            dataPerClass.clear();
            currentLabel++;
        }
        dataPerClass.push_back(data);
        if(fin.peek()==EOF)
            totalData.push_back(dataPerClass);
    }
}

/* 层次划分数据集 */
void splitDataset(vector<vector<Data>> totalData, vector<Data>& trainData, vector<Data>& testData){
    for(auto & data : totalData)
    {
        random_shuffle(data.begin(), data.end());
        trainData.insert(trainData.end(), data.begin(), data.begin() + TRAIN_SIZE_PER_CLASS);
        testData.insert(testData.end(), data.begin() + TRAIN_SIZE_PER_CLASS, data.end());
    }
    random_shuffle(trainData.begin(), trainData.end());
    random_shuffle(testData.begin(), testData.end());
}

/* 计算信息熵 Ent(D) */
double getEntropy(vector<int> sizePerClass, int total){
    double entropy = 0;
    double possibility = 0;
    for(int i=0; i<sizePerClass.size(); i++){
        possibility = sizePerClass[i]*1.0/total;
        if(!possibility)
            continue;
        entropy += (-possibility) * log(possibility) / log(2);
    }
    return entropy;
}

/* ID3策略选择增益最大的特征为最优划分 */
int chooseAttr(vector<Data> trainData, vector<int> attrSet){
    int numTrainData = trainData.size();
    vector<int> sizePerClass(CLASS_NUM, 0);
    for(int i=0; i<numTrainData; i++){  // 统计每一类样本数
        int label = trainData[i].label;
        sizePerClass[label]++;
    }
    // 虽然最大化增益熵只需求解最小化第二项即可，但为对应增益熵公式，计算Ent(D)
    double entropy = getEntropy(sizePerClass, numTrainData);

    vector<double> gain(ATTR_SIZE, INT_MIN);            // 存储选择每一个特征的增益熵，初始化为INT_MIN
    for(auto & attrIndex : attrSet)
    {
        Attr attrValueAndPerSize;
        attrValueAndPerSize.attr = attrIndex;
        for(int j=0; j<numTrainData; j++)
        {
            int attrValue = trainData[j].attr[attrIndex];
            int label = trainData[j].label;
            bool attrValueExist = false;
            for(auto & attrValueItem : attrValueAndPerSize.valueAndNum)
            {
                if(attrValue == attrValueItem.first)
                {
                    attrValueItem.second[label]++;
                    attrValueExist = true;
                    break;
                }
            }
            // 如果该attrValue还不存在在attrValueAndPerSize中，创建并插入
            if(!attrValueExist)
            {
                vector<int> perClassSize(CLASS_NUM,0);
                perClassSize[label]++;
                pair<int, vector<int>> newItem(attrValue, perClassSize);
                attrValueAndPerSize.valueAndNum.push_back(newItem);
            }
        }
        gain[attrIndex] = entropy;
        for(auto & AttrValueItem : attrValueAndPerSize.valueAndNum)
        {
            int totalNum = 0;
            for(auto & num : AttrValueItem.second)
                totalNum += num;
            gain[attrIndex] -= totalNum * 1.0 * getEntropy(AttrValueItem.second, totalNum) / numTrainData;
        }
    }
    int chosenAttrIndex = max_element(gain.begin(),gain.end()) - gain.begin();
    return chosenAttrIndex;
}

/* 判断是否停止节点分裂，即是否是叶节点；如果是，则返回<true, classLabel>*/
pair<bool,int> isLeaf(vector<Data> trainData, vector<int> attrSet)
{
    pair<bool, int> result(false,0);
    // D中样本全属于同一类C
    vector<int> sizeOfPerClass(CLASS_NUM, 0);
    for(auto & data : trainData)
        sizeOfPerClass[data.label]++;
    for(int classLabelIndex=0; classLabelIndex < sizeOfPerClass.size(); classLabelIndex++)
    {
        if(sizeOfPerClass[classLabelIndex] == trainData.size()) //同属一类
        {
            result.first=true;
            result.second=classLabelIndex;
            return result;
        }
    }

    // 属性集合为空 或者 D中样本在A上取值相同，类别标记为D中样本数最多的类
    if(attrSet.empty())
    {
        result.first=true;
        result.second= max_element(sizeOfPerClass.begin(),sizeOfPerClass.end())-sizeOfPerClass.begin();
        return result;
    }

    bool allSame = true;
    for(int i=0;i<attrSet.size();i++)
    {
        int attrIndex = attrSet[i];
        int anchor = trainData[0].attr[attrIndex];

        for(int j=1;j<trainData.size();j++)
            if(trainData[j].attr[attrIndex]!=anchor)
            {
                allSame = false;
                return result;
            }
    }
    result.first=true;
    result.second= max_element(sizeOfPerClass.begin(),sizeOfPerClass.end())-sizeOfPerClass.begin();
    return result;
}

/* 根据属性值划分训练数据给不同的孩子节点 (attrValue,(selectedData))*/
map<int, vector<Data>> dataPartition(int attrIndex, vector<Data> trainData)
{
    map<int, vector<Data>> subTrainData;
    for(int i=0;i<trainData.size();i++)
    {
        int attrValue = trainData[i].attr[attrIndex];
        subTrainData[attrValue].push_back(trainData[i]);
    }
    return subTrainData;
}

/* 节点生成 */
void treeGenerate(Node* root,vector<Data> trainData, vector<int> attrSet)
{
    pair<bool, int> isStop= isLeaf(trainData, attrSet);
    if(isStop.first)       //叶节点
    {
        root->isLeafNode= true;
        root->classLabel = isStop.second;
        return;
    }

    root->attr = chooseAttr(trainData, attrSet);

    // 从属性集中删除已划分的属性
    for(int i=0;i<attrSet.size();i++)
        if(attrSet[i]==root->attr)
        {
            attrSet.erase(attrSet.begin()+i);
            break;
        }
    map<int, vector<Data>> subTrainData = dataPartition(root->attr, trainData);
    auto it=subTrainData.begin();
    while(it!=subTrainData.end())
    {
        Node* node = new Node();
        node->attrValue = it->first;
        root->children.push_back(node);
        treeGenerate(node, it->second, attrSet);
        it++;
    }
}

/* 预测 */
int predict(Node* root, vector<int> data)
{
    Node* node = root;
    while(node!=NULL)
    {
        if(node->isLeafNode)
            return node->classLabel;
        int attrIndex = node->attr;
        for(auto branch : node->children)
        {
            if(branch->attrValue==data[attrIndex])
            {
                node=branch;
                break;
            }
        }
    }
    return -1;
}

//void printData(vector<Data> data)
//{
//    for(int i=0;i<data.size();i++){
//        for(int j=0;j<ATTR_SIZE;j++)
//            cout<<data[i].attr[j]<<" ";
//        cout<<"------"<<data[i].label<<endl;
//    }
//}

/* 使用准确率作为分类准确性的评估 */
double evaluate(Node* root,vector<Data> testData )
{
    int right = 0, wrong = 0; //区分正确的样本数和区分错误的样本数
    int groundTruth = -1, predictResult = -1;

    for(auto data : testData)
    {
        groundTruth = data.label;
        predictResult = predict(root, data.attr);
        if(groundTruth != predictResult)
            wrong++;
        else
            right++;
    }
    double accuracy = right*1.0 / (right+wrong);
    return accuracy;
}

int main()
{
//    //调试时数据文件需要放到"cmake-build-debug"文件夹下
    const string dataFileName = "iris_int.csv";
    vector<Data> trainData;
    vector<Data> testData;
    vector<vector<Data>> totalData;
    vector<int> attrSet = {0,1,2,3};
    map<int,vector<Data>> subData;
    readData(dataFileName, totalData);
    splitDataset(totalData, trainData, testData);

    Node* root = new Node();
    treeGenerate(root, trainData, attrSet);
//    int result = predict(root,{5,2,4,1});
//    cout<<"predict result="<<result<<endl;

    double accuracy = evaluate(root, testData);
    cout<<"accuracy="<<accuracy<<endl;
    return 0;
}
