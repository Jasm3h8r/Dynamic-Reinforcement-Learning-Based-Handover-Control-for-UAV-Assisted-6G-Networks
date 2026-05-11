from runner import main


if __name__ == "__main__":
    import sys

    if "--gui" not in sys.argv:
        sys.argv.append("--gui")
    main()
