
/*1. Add two integers*/
#include <stdio.h>
 void main() {
    int a = 10, b = 20;
    int c= a + b;
    printf("%d"c);
    return 0;
}   // Output: 30

/*2. Area of a circle*/
#include <studio.h>
void main() {
    float radius = 5.0;
    float area = PI * radius * radius;
    printf("Area = %.2f\n", area);
    return 0;
} // Output: Area = 78.54

/*3. Celsius to Fahrenheit*/

#include <stdio.h>
int main() {
    float celsius = 25.0;
    float fahrenheit = (celsius * 9 / 5) + 32;
    printf("Fahrenheit = %.2f\n", fahrenheit);
    return 0;
} // Output: Fahrenheit = 77.00

/*4. Swap two numbers using third variable*/
#include <stdio.h>
void main() {
    int a = 5, b = 10, temp;
    temp = a;
    a = b;
    b = temp;
    printf("After Swap: a = %d, b = %d", a, b);
    return 0;
} // Output: After Swap: a = 10, b = 5

/*5. Input five numbers and average */
#include <stdio.h>
int main() {
    int a = 10, b = 20, c = 30, d = 40, e = 50;
    float avg = (a + b + c + d + e) / 5.0;
    printf("Average = %.2f\n", avg);
    return 0;
} // Output: Average = 30.00

/*6. Square and cube of a number*/
#include <stdio.h>
int main() {
    int num = 3;
    printf("Square = %d\n", num * num);
    printf("Cube = %d\n", num * num * num);
    return 0;
} // Output: Square = 9, Cube = 27

/*7. Minutes to hours and minutes*/

#include <stdio.h>
int main() {
    int minutes = 135;
    int hours = minutes / 60;
    int rem = minutes % 60;
    printf("%d minutes = %d hour(s) and %d minute(s)\n", minutes, hours, rem);
    return 0;
} // Output: 135 minutes = 2 hour(s) and 15 minute(s)

/*8. Perimeter of rectangle*/

#include <stdio.h>
int main() {
    int length = 10, width = 5;
    int perimeter = 2 * (length + width);
    printf("Perimeter = %d", perimeter);
    return 0;
} // Output: Perimeter = 30

/*9. Area of a triangle*/
#include <stdio.h>
int main() {
    float base = 10, height = 5;
    float area = 0.5 * base * height;
    printf("Area = %.2f\n", area);
    return 0;
} // Output: Area = 25

/*10. Total marks and percentage*/
#include <stdio.h>
int main() {
    int math1 = 85, eng = 90, marathi = 78, sci = 88, math2 = 92;
    int total = math1 + eng + marathi + sci + math2;
    float percentage = total / 5.0;
    printf("Total = %d\n", total);
    printf("Percentage = %.2f%%\n", percentage);
    return 0;
} // Output: Total = 433, Percentage = 86.60%
