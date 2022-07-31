from django.db import models


class Features(models.Model):
    ClosedDate = models.FloatField(default=0)
    LoanCurrentDaysDelinquent = models.FloatField(default=0)
    LoanMonthsSinceOrigination = models.FloatField(default=0)
    LP_CustomerPrincipalPayments = models.FloatField(default=0)
    LP_GrossPrincipalLoss = models.FloatField(default=0)
    LP_NetPrincipalLoss = models.FloatField(default=0)
    LP_CustomerPayments = models.FloatField(default=0)
    EmploymentStatus = models.FloatField(default=0)
    LP_ServiceFees = models.FloatField(default=0)
    LoanOriginalAmount = models.FloatField(default=0)
    Investors = models.FloatField(default=0)
    EstimatedReturn = models.FloatField(default=0)
    LP_InterestandFees = models.FloatField(default=0)
    MonthlyLoanPayment = models.FloatField(default=0)
    LP_CollectionFees = models.FloatField(default=0)
    EstimatedEffectiveYield = models.FloatField(default=0)
    EstimatedLoss = models.FloatField(default=0)
    Term = models.FloatField(default=0)
    BorrowerAPR = models.FloatField(default=0)
    LP_NonPrincipalRecoverypayments = models.FloatField(default=0)
    BorrowerRate = models.FloatField(default=0)
    ListingCategory = models.FloatField(default=0)
    LenderYield = models.FloatField(default=0)
    CreditScoreRangeUpper = models.FloatField(default=0)
    OpenRevolvingMonthlyPayment = models.FloatField(default=0)
    ProsperScore = models.FloatField(default=0)
    CreditScoreRangeLower = models.FloatField(default=0)
    RevolvingCreditBalance = models.FloatField(default=0)
    ProsperRating = models.FloatField(default=0)
    AvailableBankcardCredit = models.FloatField(default=0)
    EmploymentStatusDuration = models.FloatField(default=0)
    DebtToIncomeRatio = models.FloatField(default=0)
    StatedMonthlyIncome = models.FloatField(default=0)
    BankcardUtilization = models.FloatField(default=0)
    TotalCreditLinespast7years = models.FloatField(default=0)
    TotalTrades = models.FloatField(default=0)
    
    
    def __str__(self):
        return str(self.id)



