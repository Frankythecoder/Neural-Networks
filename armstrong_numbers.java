public class ArmstrongNumbers {
    public static void main(String[] args) {
        System.out.println("Armstrong numbers from 1 to 100:");
        for (int num = 1; num <= 100; num++) {
            if (isArmstrong(num)) {
                System.out.print(num + " ");
            }
        }
    }

    public static boolean isArmstrong(int number) {
        int originalNumber, remainder, result = 0;
        originalNumber = number;

        while (originalNumber != 0) {
            remainder = originalNumber % 10;
            result += Math.pow(remainder, 3);
            originalNumber /= 10;
        }

        return result == number;
    }
}
