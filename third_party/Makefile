export GTEST_OUT=$(OUT)/gtest

gtest: export CXX=$(CC)
gtest: export CXXFLAGS=$(CFLAGS)
gtest: | $(GTEST_OUT)
	$(MAKE) -C gtest-1.7.0/ $(GTEST_OUT)/gtest_main.a

clean:
	$(MAKE) -C gtest-1.7.0/make/ clean

$(GTEST_OUT):
	mkdir -p $(OUT)/gtest
