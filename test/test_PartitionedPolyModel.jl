function test_Partition()
    limits = [-1.0 1.0; -1.0 1.0]
    prt = Partition(limits)
    split!(prt, 1, 1)
    split!(prt, 2, 2)

    @test prt.regions[1].limits == [-1.0 0.0; -1.0 1.0]
    @test prt.regions[1].key == 1
    @test prt.regions[2].limits == [0.0 1.0; -1.0 0.0]
    @test prt.regions[2].key == 2
    @test prt.regions[3].limits == [0.0 1.0; 0.0 1.0]
    @test prt.regions[3].key == 3

    @test prt.space.left.limits == [-1.0 0.0; -1.0 1.0]
    @test prt.space.right.left.limits == [0.0 1.0; -1.0 0.0]
    @test prt.space.right.right.limits == [0.0 1.0; 0.0 1.0]

    X = [-1.0 1.0; 0.0 -1.0; 1.0 1.0]
    @test locate(prt, X) == [1, 2, 3]
end


test_Partition()