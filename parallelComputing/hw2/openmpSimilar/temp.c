#include <stdio.h>
#include <stdlib.h>

//append two strings and return the third
 char *strcat(char *str1, char *str2) {
  char *temp1;
  char *temp2, *combined;
  int totalSize;

  //get combined size of strings
  totalSize = 0;

  temp1 = str1;
  while (*temp1 != '\0') {
    totalSize++;
    temp1++;
  }
  
  temp1 = str2;
  while (*temp1 != '\0') {
    totalSize++;
    temp1++;
  }

  //allocate memory for combined string
  combined = (char*) malloc(sizeof(char) * totalSize);

  //copy string 1
  temp1 = str1;
  temp2 = combined;
  while (*temp1 != '\0') {
    *temp2 = *temp1;
    temp2++;
    temp1++;
  }

  //copy string2
  temp1 = str2;
  while (*temp1 != '\0') {
    *temp2 = *temp1;
    temp2++;
    temp1++;
  }

  return combined;
}

char *toStr4mNum(int a) {
  char *str1, *str2;
  char *temp1, *temp2;

  str1 = (char *) malloc(10);
  str2 = (char *) malloc(10);

  temp1 = str1;

  while ( a != 0) {
    *temp1 = (char)((int)('0') + (a % 10));
    a = a/10;
    temp1++;
  }

  temp2 = str2;
  temp1--;
  while (temp1 != str1) {
    *temp2++ = *temp1--;
  }
  
  *temp2++ = *temp1;
  *temp2 = '\0';
  
  return str2;
}


int main(int argc, char **argv) {

  char* str1;
  char* str2;
  char *str3;
  int num;
  str1 = (char*) malloc(10);
  str1 = "hello";
  str2 = (char*) malloc(10);
  str2 = "world";
  str3 = strcat(str1, str2);

  num = 10;
  printf("\n%s\n", str1);
  printf("\n%s\n", str2);
  printf("\n%s\n", str3);
  
  str2 = (char *) toStr4mNum(num);
  printf("\n%s\n", str2);
  return 0;
}
