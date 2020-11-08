TEST(ArgMaxTest, UnsupportedType_NEG) {
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1, 1, 2, 4}, {
                                                                             1,
                                                                             2,
                                                                             7,
                                                                             8,
                                                                             1,
                                                                             9,
                                                                             7,
                                                                             3,
                                                                         });
  Tensor dimension_tensor = makeInputTensor<DataType::S32>({}, {3});
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  ArgMaxParams params{};
  params.output_type = DataType::U8;
  ArgMax kernel(&input_tensor, &dimension_tensor, &output_tensor, params);
  kernel.configure();
  EXPECT_ANY_THROW(kernel.execute());
}
