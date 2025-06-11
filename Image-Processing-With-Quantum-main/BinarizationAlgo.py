grey_img = [[0,3],[5,2]]
res_img = []

def processNumElems(mat1, mat2):
    for rows in mat1:
        tempRows = []
        for elems in rows:
            tempRows.append(f"{elems:08b}")
        mat2.append(tempRows)

def displayValuesOfMatrix(mat):
    for i in mat:
        for j in i:
            print(j)

def processBinaryElems(mat):
    firstRow = mat[0]
    secondRow = mat[1]
    for i in firstRow:
        for j in secondRow:
            e1 = str(i).split()
            print(e1) 
    

def binarization():
    ...            

processNumElems(grey_img, res_img)
displayValuesOfMatrix(res_img)
print(" ")
processBinaryElems(res_img)