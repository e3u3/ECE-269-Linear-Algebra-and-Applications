% --------------- MOVIELENS DATA ---------------
A = csvread('ratings.csv',1,0)
userID = A(1:length(A),1)
movieID = A(1:length(A),2)
rating = A(1:length(A),3)

uniqueUser = unique(userID)
uniqueMovie = unique(movieID)
ACov = cov(A)

% Calculate the average rating over all of the data.
ratingSum = 0;
for i = 1:length(rating)
    ratingSum = rating(i) + ratingSum ;
end
avgRating = ratingSum / length(rating);

% If a rating has a 0 value, replace it iwth the average rating, in order
to account for A being incomplete.
for i = 1:length(rating)
    if(rating(i) == 0)
        rating(i) = avgRating;
    end
end

% Calculate the SVD of matrix A.
[U,S,V] = svd(A,'econ')

M = 100,004, N = 3, K <= 3

% Assign P to be U*S from the SVD
P = U*S
% Assign Q To be V from the SVD
Q = V

pi = P(1,1:3)
qj = Q(1,1:3)
rHat = dot(transpose(pi),qj)

% --------------- JESTER DATA ---------------
% LATENT FACTOR MODEL - 
A = xlsread('jester-data-1.xls');

for i = 1:length(A)
    for j = 1:100
        if A(i,j) == 99
            A(i,j) = 0;
        end
    end
end

trainData = A(1:20000,:);
testData = A(20001:24983,:);

tic
% Calculate the SVD of matrix trainData.
[U,S,V] = svd(trainData,'econ');
% Assign P to be U*S from the SVD
trainP = U*S;
% Assign Q To be V from the SVD
trainQ = V;

% Calculate the SVD of matrix testData.
[U,S,V] = svd(testData,'econ');
% Assign P to be U*S from the SVD
testP = U*S;
% Assign Q To be V from the SVD
testQ = V;

% Calculate predicted joke ratings for currentUser in testData. These
% ratings will be loaded into an array where each joke rating will be in
% the index of that joke number. This can be repeated for any user that is
% in the testData.
currentUser = 3;
rHat = zeros(100,1);
for i = 1:100
    pi = testP(currentUser,1:100);
    qj = trainQ(i,1:100);
    rHat(i) = dot(transpose(pi),qj);
end
toc

% NON-PERSONALIZED MODEL -
tic
% Calculate the average joke rating for each column to create a
% non-personalized model.
averageRatings = zeros(100,1);
for i = 1:100
    currentSum = 0;
    for j = 1:length(trainData)
        currentSum = currentSum + trainData(j,i);
    end
    averageRatings(i) = currentSum/length(trainData);
end
toc

