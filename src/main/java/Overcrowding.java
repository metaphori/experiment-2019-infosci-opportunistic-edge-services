
public enum Overcrowding {
    NONE,
    AT_RISK,
    OVERCROWDED;

    public static Overcrowding overcrowded() { return OVERCROWDED; }
    public static Overcrowding atRisk() { return AT_RISK; }
    public static Overcrowding none() { return NONE; }
    
}

